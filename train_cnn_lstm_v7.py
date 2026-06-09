from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    TimeDistributed,
    Conv1D,
    BatchNormalization,
    ReLU,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.cv import stratified_group_kfold_split
from src.dataset_builder import (
    build_dataset_from_catalog,
    pack_windows_and_stats_into_sequences,
)
from src.preprocess import fit_standard_scaler, transform_with_scaler


TARGET_LABELS = [
    "healthy",
    "bearing bpfi",
    "bearing bpfo",
    "bearing pump",
    "impeller",
]

TARGET_LABELS_NAME = "fault_type_motor2"

TARGET_LABELS_NAME = "bearing_pump_subtypes"
OUTPUT_DIR = Path(f"outputs/{TARGET_LABELS_NAME}_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CONFIG
# =========================
BATCH_SIZE = 2
EPOCHS = 40
FFT_BATCH_SIZE = 256


def filter_df_by_labels(
    df: pd.DataFrame,
    labels: list[str],
    label_col: str = "falha",
) -> pd.DataFrame:
    df = df.copy()

    original_labels = df[label_col].astype(str)
    df = df[original_labels.isin(labels)].copy()

    if df.empty:
        raise ValueError(f"Nenhuma linha encontrada para labels: {labels}")

    found = sorted(df[label_col].astype(str).unique().tolist())
    missing = sorted(set(labels) - set(found))

    print("\nFiltro direto ativado")
    print("Labels alvo:", labels)
    print("Labels encontrados:", found)
    if missing:
        print("Labels ausentes no dataset filtrado:", missing)

    return df


# =========================
# DATASET
# =========================
def load_filtered_dataset() -> pd.DataFrame:
    df = pd.read_csv("outputs/catalog_samples.csv")

    df = df[
        (df["sensor"] == "Vibration") &
        (df["equipamento"] == "Motor-2")
    ].copy()

    if "is_sample_valid" in df.columns:
        df = df[df["is_sample_valid"] == True].copy()

    if "n_channels_found" in df.columns:
        df = df[df["n_channels_found"] == 5].copy()

    group_counts = df.groupby("falha")["grupo_condicao"].nunique()
    valid_labels = group_counts[group_counts >= 2].index
    df = df[df["falha"].isin(valid_labels)].copy()

    if df.empty:
        raise ValueError("Dataset filtrado ficou vazio.")

    return df


# =========================
# HELPERS
# =========================
def split_train_val_by_group(
    train_df: pd.DataFrame,
    label_col: str = "falha",
    group_col: str = "grupo_condicao",
    val_size: float = 0.2,
    random_state: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_df = (
        train_df[[group_col, label_col]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    n_groups = len(group_df)
    n_classes = group_df[label_col].nunique()

    if isinstance(val_size, float):
        n_val = int(np.ceil(n_groups * val_size))
    else:
        n_val = int(val_size)

    label_counts = group_df[label_col].value_counts()
    can_stratify = (
        label_counts.min() >= 2
        and n_groups >= 2
        and n_val >= n_classes
    )

    if can_stratify:
        train_groups, val_groups = train_test_split(
            group_df[group_col],
            test_size=val_size,
            random_state=random_state,
            stratify=group_df[label_col],
        )
    else:
        print("\n[WARN] Split interno sem estratificação: grupos insuficientes por classe.")
        train_groups, val_groups = train_test_split(
            group_df[group_col],
            test_size=val_size,
            random_state=random_state,
            stratify=None,
        )

    train_inner_df = train_df[train_df[group_col].isin(train_groups)].copy()
    val_df = train_df[train_df[group_col].isin(val_groups)].copy()

    return train_inner_df, val_df


def encode_with_train_only(
    y_train_raw: np.ndarray,
    y_val_raw: np.ndarray,
    y_test_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    train_labels = set(np.unique(y_train_raw))
    val_labels = set(np.unique(y_val_raw))
    test_labels = set(np.unique(y_test_raw))

    unseen_val = sorted(val_labels - train_labels)
    unseen_test = sorted(test_labels - train_labels)

    if unseen_val or unseen_test:
        print("\n[WARN] Labels ausentes no treino.")
        if unseen_val:
            print("[WARN] Val com labels fora do treino:", unseen_val)
        if unseen_test:
            print("[WARN] Test com labels fora do treino:", unseen_test)

    all_labels = sorted(train_labels | val_labels | test_labels)

    encoder = LabelEncoder()
    encoder.fit(all_labels)

    y_train = encoder.transform(y_train_raw)
    y_val = encoder.transform(y_val_raw)
    y_test = encoder.transform(y_test_raw)

    return y_train, y_val, y_test, encoder


def fit_transform_stats_sequence_train_val_test(
    X_train_stats_seq: np.ndarray,
    X_val_stats_seq: np.ndarray,
    X_test_stats_seq: np.ndarray,
):
    if X_train_stats_seq.ndim != 3 or X_val_stats_seq.ndim != 3 or X_test_stats_seq.ndim != 3:
        raise ValueError(
            "Esperado arrays 3D para stats por sequência. "
            f"Recebido train={X_train_stats_seq.shape}, "
            f"val={X_val_stats_seq.shape}, test={X_test_stats_seq.shape}"
        )

    train_shape = X_train_stats_seq.shape
    val_shape = X_val_stats_seq.shape
    test_shape = X_test_stats_seq.shape

    X_train_2d = X_train_stats_seq.reshape(-1, train_shape[-1])
    X_val_2d = X_val_stats_seq.reshape(-1, val_shape[-1])
    X_test_2d = X_test_stats_seq.reshape(-1, test_shape[-1])

    scaler = StandardScaler()
    scaler.fit(X_train_2d)

    X_train_scaled = scaler.transform(X_train_2d).reshape(train_shape).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_2d).reshape(val_shape).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_2d).reshape(test_shape).astype(np.float32)

    print("[Scaling Stats Seq] Concluído.")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def fit_transform_sequence_train_val_test(
    X_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    X_test_seq: np.ndarray,
):
    if X_train_seq.ndim != 4 or X_val_seq.ndim != 4 or X_test_seq.ndim != 4:
        raise ValueError(
            "Esperado arrays 4D em treino/val/teste. "
            f"Recebido train={X_train_seq.shape}, val={X_val_seq.shape}, test={X_test_seq.shape}"
        )

    train_shape = X_train_seq.shape
    val_shape = X_val_seq.shape
    test_shape = X_test_seq.shape

    print("\n[Scaling Signal] Preparando reshape 4D -> 3D...")
    X_train_flat = X_train_seq.reshape(-1, train_shape[2], train_shape[3])
    X_val_flat = X_val_seq.reshape(-1, val_shape[2], val_shape[3])
    X_test_flat = X_test_seq.reshape(-1, test_shape[2], test_shape[3])

    print(f"[Scaling Signal] Train flat: {X_train_flat.shape}")
    print(f"[Scaling Signal] Val flat:   {X_val_flat.shape}")
    print(f"[Scaling Signal] Test flat:  {X_test_flat.shape}")

    scaler = fit_standard_scaler(X_train_flat)

    X_train_scaled = transform_with_scaler(X_train_flat, scaler).reshape(train_shape).astype(np.float32)
    X_val_scaled = transform_with_scaler(X_val_flat, scaler).reshape(val_shape).astype(np.float32)
    X_test_scaled = transform_with_scaler(X_test_flat, scaler).reshape(test_shape).astype(np.float32)

    print("[Scaling Signal] Concluído.")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def generate_fft_features_batched(X_win: np.ndarray, batch_size: int = 256) -> np.ndarray:
    if X_win.ndim != 3:
        raise ValueError(f"Esperado X_win 3D para FFT, recebido shape {X_win.shape}")

    print("\n[FFT] Gerando features de frequência por lotes...")
    print(f"[FFT] Shape antes: {X_win.shape}")
    print(f"[FFT] Batch size: {batch_size}")

    fft_batches = []

    for start in tqdm(range(0, len(X_win), batch_size), desc="FFT por lotes"):
        end = start + batch_size
        X_chunk = X_win[start:end]

        X_fft_chunk = np.abs(np.fft.fft(X_chunk, axis=1)).astype(np.float32)
        X_fft_chunk = np.log1p(X_fft_chunk)

        fft_batches.append(X_fft_chunk)

    X_fft = np.concatenate(fft_batches, axis=0).astype(np.float32)
    print(f"[FFT] Shape FFT: {X_fft.shape}")
    return X_fft


# =========================
# MODEL
# =========================
def build_cnn_feature_extractor(window_size: int, n_features: int) -> tf.keras.Model:
    inputs = Input(shape=(window_size, n_features), name="window_input")

    x = Conv1D(32, 9, padding="same", use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.15)(x)

    x = Conv1D(64, 7, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.20)(x)

    x = Conv1D(128, 5, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.25)(x)

    x = Conv1D(128, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(96, activation="relu")(x)
    x = Dropout(0.30)(x)

    return Model(inputs, x, name="cnn_feature_extractor_v6")


def build_cnn_gap_v0_fusion_model(
    seq_len: int,
    window_size: int,
    n_features: int,
    n_stats_window: int,
    n_classes: int,
) -> tf.keras.Model:
    cnn_extractor = build_cnn_feature_extractor(window_size, n_features)

    signal_input = Input(shape=(seq_len, window_size, n_features), name="signal_input")
    stats_seq_input = Input(shape=(seq_len, n_stats_window), name="stats_seq_input")

    x_signal = TimeDistributed(cnn_extractor, name="td_cnn")(signal_input)
    x_signal = GlobalAveragePooling1D(name="temporal_gap_signal")(x_signal)

    x_stats = TimeDistributed(Dense(64, activation="relu"), name="td_stats_dense_1")(stats_seq_input)
    x_stats = Dropout(0.15, name="stats_dropout_1")(x_stats)
    x_stats = TimeDistributed(Dense(64, activation="relu"), name="td_stats_dense_2")(x_stats)
    x_stats = GlobalAveragePooling1D(name="temporal_gap_stats")(x_stats)

    gate = Dense(
        x_signal.shape[-1],
        activation="sigmoid",
        name="stats_gate"
    )(x_stats)

    x_signal_gated = tf.keras.layers.Multiply(name="apply_stats_gate")([x_signal, gate])

    x = Concatenate(axis=-1, name="concat_signal_stats")([x_signal_gated, x_stats])

    x = Dense(64, activation="relu", name="fusion_dense_1")(x)
    x = Dropout(0.35, name="fusion_dropout_1")(x)

    x = Dense(32, activation="relu", name="fusion_dense_2")(x)
    x = Dropout(0.30, name="fusion_dropout_2")(x)

    outputs = Dense(n_classes, activation="softmax", name="classifier")(x)

    model = Model(
        inputs=[signal_input, stats_seq_input],
        outputs=outputs,
        name="cnn_gap_v0_windowstats_gated_fusion_model_v6",
    )

    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =========================
# PLOTS
# =========================
def plot_training_history(history, output_dir: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Loss durante treino")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend()
    plt.title("Accuracy durante treino")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], output_dir: Path):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm_norm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão Normalizada")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_normalized.png", dpi=150)
    plt.close()


class EpochDiagnosticCallback(tf.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        history = self.model.history.history

        train_loss = history.get("loss", [])
        val_loss = history.get("val_loss", [])
        train_acc = history.get("accuracy", [])
        val_acc = history.get("val_accuracy", [])

        if not val_loss:
            print("\n[DIAG] sem histórico de validação.")
            return

        best_epoch = int(np.argmin(val_loss))
        last_epoch = len(val_loss) - 1

        print("\n" + "=" * 60)
        print("[DIAG] RESUMO FINAL DO TREINO")
        print("=" * 60)
        print(f"Última época executada: {last_epoch + 1}")
        print(f"Melhor época (menor val_loss): {best_epoch + 1}")
        print(f"train_loss final: {train_loss[last_epoch]:.4f}")
        print(f"val_loss final:   {val_loss[last_epoch]:.4f}")
        print(f"train_acc final:  {train_acc[last_epoch]:.4f}")
        print(f"val_acc final:    {val_acc[last_epoch]:.4f}")

        gap_acc = train_acc[last_epoch] - val_acc[last_epoch]
        gap_loss = val_loss[last_epoch] - train_loss[last_epoch]

        print(f"gap_acc final:    {gap_acc:.4f}")
        print(f"gap_loss final:   {gap_loss:.4f}")

        if last_epoch > best_epoch:
            print(f"[DIAG] treino passou do melhor ponto por {last_epoch - best_epoch} época(s).")
            print("[DIAG] motivo provável: overfitting / perda de generalização.")
        else:
            print("[DIAG] treino terminou no melhor ponto observado.")

        if gap_acc > 0.15 and gap_loss > 0.5:
            print("[DIAG] sinal forte de overfitting.")
        elif val_acc[last_epoch] < 0.05:
            print("[DIAG] modelo com baixa generalização.")
        else:
            print("[DIAG] generalização aceitável dentro do cenário atual.")

        print("=" * 60)


def build_sequence_split_v6(
    df_subset: pd.DataFrame,
    label_col: str,
    window_size: int,
    overlap: float,
    drop_last: bool,
    nrows: int | None,
    seq_len: int,
    sequence_stride: int | None,
    drop_incomplete_sequences: bool,
    grouping_col: str,
    stats_include_optional: bool = False,
):
    X_win, X_stats_win, y_raw, meta_df, stats_feature_names = build_dataset_from_catalog(
        df_subset=df_subset,
        label_col=label_col,
        selected_columns=None,
        window_size=window_size,
        overlap=overlap,
        drop_last=drop_last,
        nrows=nrows,
        return_stats=True,
        stats_include_optional=stats_include_optional,
        stats_mode="v0_plus_spectral",
        include_band_energy=False,
    )

    #X_fft = generate_fft_features_batched(X_win, batch_size=FFT_BATCH_SIZE)
    #X_win = np.concatenate([X_win, X_fft], axis=2).astype(np.float32)
    #del X_fft

    print(f"[FFT] Novo shape após concat: {X_win.shape}")
    print(f"[Stats] Shape stats por janela: {X_stats_win.shape}")

    X_seq, X_stats_seq, y_seq_raw, meta_seq = pack_windows_and_stats_into_sequences(
        X=X_win,
        X_stats=X_stats_win,
        y=y_raw,
        meta_df=meta_df,
        seq_len=seq_len,
        sequence_stride=sequence_stride,
        drop_incomplete=drop_incomplete_sequences,
        grouping_col=grouping_col,
    )

    print(f"[Stats] Shape stats por sequência: {X_stats_seq.shape}")

    return X_seq, X_stats_seq, y_seq_raw, meta_seq, stats_feature_names


# =========================
# MAIN
# =========================
def main():
    global_start = time.time()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    print("GPUs disponíveis:", tf.config.list_physical_devices("GPU"))

    label_col = "falha_base"
    window_size = 1024
    overlap = 0.5
    drop_last = True
    nrows = None
    seq_len = 10
    train_sequence_stride = 10
    eval_sequence_stride = 10
    drop_incomplete_sequences = True
    grouping_col = "grupo_condicao"

    print("Carregando dataset filtrado...")
    df = load_filtered_dataset()

    # =========================
    # NORMALIZA LABEL (REMOVE 1,2,3)
    # =========================
    df['falha_base'] = df['falha'].str.replace(r'\s+\d+$', '', regex=True)

    print("\n[DEBUG] labels normalizados:")
    print(sorted(df['falha_base'].unique()))

    print("\n[DEBUG] contagem por classe:")
    print(df['falha_base'].value_counts())

    df = filter_df_by_labels(
        df,
        labels=TARGET_LABELS,
        label_col=label_col,
    )

    print("Shape dataset filtrado:", df.shape)
    print("Classes:", df[label_col].nunique())
    print("Grupos:", df["grupo_condicao"].nunique())

    outer_splits = stratified_group_kfold_split(
        df=df,
        group_col="grupo_condicao",
        label_col=label_col,
        n_splits=3,
        random_state=42,
    )
    train_df, test_df = outer_splits[0]

    print("\nResumo do split externo:")
    print("train_df shape:", train_df.shape)
    print("test_df shape:", test_df.shape)
    print("train groups:", train_df["grupo_condicao"].nunique())
    print("test groups:", test_df["grupo_condicao"].nunique())
    print("train labels:", train_df[label_col].nunique())
    print("test labels:", test_df[label_col].nunique())

    train_inner_df, val_df = split_train_val_by_group(
        train_df=train_df,
        label_col=label_col,
        group_col="grupo_condicao",
        val_size=0.2,
        random_state=123,
    )

    print("\nResumo do split interno:")
    print("train_inner_df shape:", train_inner_df.shape)
    print("val_df shape:", val_df.shape)
    print("train_inner groups:", train_inner_df["grupo_condicao"].nunique())
    print("val groups:", val_df["grupo_condicao"].nunique())
    print("train_inner labels:", train_inner_df[label_col].nunique())
    print("val labels:", val_df[label_col].nunique())

    print("\n[Build] Montando sequências de treino...")
    X_train_raw, X_train_stats_seq_raw, y_train_raw, meta_train, stats_feature_names = build_sequence_split_v6(
        df_subset=train_inner_df,
        label_col=label_col,
        window_size=window_size,
        overlap=overlap,
        drop_last=drop_last,
        nrows=nrows,
        seq_len=seq_len,
        sequence_stride=train_sequence_stride,
        drop_incomplete_sequences=drop_incomplete_sequences,
        grouping_col=grouping_col,
        stats_include_optional=False,
    )

    print("\n[Build] Montando sequências de validação...")
    X_val_raw, X_val_stats_seq_raw, y_val_raw, meta_val, _ = build_sequence_split_v6(
        df_subset=val_df,
        label_col=label_col,
        window_size=window_size,
        overlap=overlap,
        drop_last=drop_last,
        nrows=nrows,
        seq_len=seq_len,
        sequence_stride=eval_sequence_stride,
        drop_incomplete_sequences=drop_incomplete_sequences,
        grouping_col=grouping_col,
        stats_include_optional=False,
    )

    print("\n[Build] Montando sequências de teste...")
    X_test_raw, X_test_stats_seq_raw, y_test_raw, meta_test, _ = build_sequence_split_v6(
        df_subset=test_df,
        label_col=label_col,
        window_size=window_size,
        overlap=overlap,
        drop_last=drop_last,
        nrows=nrows,
        seq_len=seq_len,
        sequence_stride=eval_sequence_stride,
        drop_incomplete_sequences=drop_incomplete_sequences,
        grouping_col=grouping_col,
        stats_include_optional=False,
    )

    print("\nShapes brutos:")
    print("X_train_raw:", X_train_raw.shape)
    print("X_train_stats_seq_raw:", X_train_stats_seq_raw.shape)
    print("X_val_raw:", X_val_raw.shape)
    print("X_val_stats_seq_raw:", X_val_stats_seq_raw.shape)
    print("X_test_raw:", X_test_raw.shape)
    print("X_test_stats_seq_raw:", X_test_stats_seq_raw.shape)

    np.savez(
        OUTPUT_DIR / "preprocessed_dataset.npz",
        X_train_stats_seq_raw=X_train_stats_seq_raw,
        X_val_stats_seq_raw=X_val_stats_seq_raw,
        X_test_stats_seq_raw=X_test_stats_seq_raw,
        y_train=y_train_raw,
        y_val=y_val_raw,
        y_test=y_test_raw,
    )

    diag_dir = OUTPUT_DIR / "diagnostics_pre_mlp"
    diag_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[DIAG PRE-MLP] Salvando em: {diag_dir}")

    # 1) contagem por classe
    df[label_col].value_counts().rename_axis("classe").reset_index(name="n_linhas").to_csv(
        diag_dir / "class_distribution.csv", index=False
    )

    # 2) grupos por classe
    df.groupby(label_col)["grupo_condicao"].nunique().rename("n_grupos").reset_index().to_csv(
        diag_dir / "group_distribution.csv", index=False
    )

    # 3) grupos de cada split
    train_inner_df[["grupo_condicao", label_col]].drop_duplicates().to_csv(
        diag_dir / "split_train_groups.csv", index=False
    )
    val_df[["grupo_condicao", label_col]].drop_duplicates().to_csv(
        diag_dir / "split_val_groups.csv", index=False
    )
    test_df[["grupo_condicao", label_col]].drop_duplicates().to_csv(
        diag_dir / "split_test_groups.csv", index=False
    )

    # 4) interseção entre splits
    train_groups = set(train_inner_df["grupo_condicao"].unique())
    val_groups = set(val_df["grupo_condicao"].unique())
    test_groups = set(test_df["grupo_condicao"].unique())

    pd.DataFrame([
        {"intersection": "train_val", "n_common": len(train_groups & val_groups)},
        {"intersection": "train_test", "n_common": len(train_groups & test_groups)},
        {"intersection": "val_test", "n_common": len(val_groups & test_groups)},
    ]).to_csv(diag_dir / "split_intersections.csv", index=False)

    # 5) rpm por classe
    if "rpm" in df.columns:
        df.groupby([label_col, "rpm"]).size().rename("n_linhas").reset_index().to_csv(
            diag_dir / "rpm_by_class.csv", index=False
        )

    print("[DIAG PRE-MLP] OK")

    return

    print("[DIAG PRE-MLP] OK")
    print("Encerrando execução aqui.")
    exit()
    
    print("✔ dataset tabular salvo")

    print("stats mean:", np.mean(X_train_stats_seq_raw))
    print("stats std:", np.std(X_train_stats_seq_raw))

    y_train, y_val, y_test, label_encoder = encode_with_train_only(
        y_train_raw, y_val_raw, y_test_raw
    )

#

    pd.Series(y_train_raw, name="classe").value_counts().rename_axis("classe").reset_index(name="n_sequences").to_csv(
    diag_dir / "train_sequence_distribution.csv", index=False
    )

    pd.Series(y_val_raw, name="classe").value_counts().rename_axis("classe").reset_index(name="n_sequences").to_csv(
        diag_dir / "val_sequence_distribution.csv", index=False
    )

    pd.Series(y_test_raw, name="classe").value_counts().rename_axis("classe").reset_index(name="n_sequences").to_csv(
        diag_dir / "test_sequence_distribution.csv", index=False
    )

#

    print("stats mean:", np.mean(X_train_stats_seq_raw))
    print("stats std:", np.std(X_train_stats_seq_raw))

    from sklearn.ensemble import RandomForestClassifier

    X_train_rf = X_train_stats_seq_raw.mean(axis=1)
    X_val_rf = X_val_stats_seq_raw.mean(axis=1)

    print("RF input shape:", X_train_rf.shape)

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train_rf, y_train)

    val_score = rf.score(X_val_rf, y_val)
    print("RF VAL ACC:", val_score)

    y_train, y_val, y_test, label_encoder = encode_with_train_only(
        y_train_raw=y_train_raw,
        y_val_raw=y_val_raw,
        y_test_raw=y_test_raw,
    )
    classes = [str(c) for c in label_encoder.classes_]

    X_train, X_val, X_test, signal_scaler = fit_transform_sequence_train_val_test(
        X_train_seq=X_train_raw,
        X_val_seq=X_val_raw,
        X_test_seq=X_test_raw,
    )

    X_train_stats_seq, X_val_stats_seq, X_test_stats_seq, stats_scaler = fit_transform_stats_sequence_train_val_test(
        X_train_stats_seq=X_train_stats_seq_raw,
        X_val_stats_seq=X_val_stats_seq_raw,
        X_test_stats_seq=X_test_stats_seq_raw,
    )

    print("\nShapes finais:")
    print("X_train:", X_train.shape)
    print("X_train_stats_seq:", X_train_stats_seq.shape)
    print("X_val:", X_val.shape)
    print("X_val_stats_seq:", X_val_stats_seq.shape)
    print("X_test:", X_test.shape)
    print("X_test_stats_seq:", X_test_stats_seq.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    print("y_test:", y_test.shape)

    model = build_cnn_gap_v0_fusion_model(
        seq_len=X_train.shape[1],
        window_size=X_train.shape[2],
        n_features=X_train.shape[3],
        n_stats_window=X_train_stats_seq.shape[2],
        n_classes=len(classes),
    )
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=40,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        EpochDiagnosticCallback(),
    ]

    print("\nTreinando modelo V6...")
    train_start = time.time()

    history = model.fit(
        [X_train, X_train_stats_seq],
        y_train,
        validation_data=([X_val, X_val_stats_seq], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    train_elapsed = time.time() - train_start
    print(f"\nTempo de treino: {train_elapsed:.2f} s ({train_elapsed / 60:.2f} min)")

    print("\nGerando previsões no teste...")
    predict_start = time.time()

    with tf.device("/CPU:0"):
        y_pred_proba = model.predict([X_test, X_test_stats_seq], batch_size=4, verbose=1)

    predict_elapsed = time.time() - predict_start
    print(f"Tempo de predição: {predict_elapsed:.2f} s ({predict_elapsed / 60:.2f} min)")

    y_pred = np.argmax(y_pred_proba, axis=1)

    # =========================
    # AVALIAÇÃO POR JANELA
    # =========================
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(OUTPUT_DIR / "classification_report_windows.csv", index=True)

    print("\nClassification Report (janelas):")
    print(classification_report(
        y_test,
        y_pred,
        target_names=classes,
        zero_division=0,
    ))

    cm_windows = confusion_matrix(y_test, y_pred)
    np.save(OUTPUT_DIR / "confusion_matrix_windows.npy", cm_windows)

    plot_training_history(history, OUTPUT_DIR)
    plot_confusion_matrix(cm_windows, classes, OUTPUT_DIR)

    # =========================
    # AVALIAÇÃO POR GRUPO
    # =========================
    df_pred = meta_test.copy().reset_index(drop=True)
    df_pred["y_true"] = y_test
    df_pred["y_pred"] = y_pred

    grouped = df_pred.groupby("grupo_condicao").agg({
        "y_true": lambda x: x.mode()[0],
        "y_pred": lambda x: x.value_counts().idxmax(),
    })

    cm_group = confusion_matrix(grouped["y_true"], grouped["y_pred"])
    np.save(OUTPUT_DIR / "confusion_matrix_group.npy", cm_group)

    group_report_dict = classification_report(
        grouped["y_true"],
        grouped["y_pred"],
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    group_report_df = pd.DataFrame(group_report_dict).transpose()
    group_report_df.to_csv(OUTPUT_DIR / "classification_report_group.csv", index=True)

    print("\nClassification Report (grupo_condicao):")
    print(classification_report(
        grouped["y_true"],
        grouped["y_pred"],
        target_names=classes,
        zero_division=0,
    ))

    model.save(OUTPUT_DIR / "cnn_gap_v6.keras")

    total_elapsed = time.time() - global_start

    summary = {
        "n_train_sequences": int(X_train.shape[0]),
        "n_val_sequences": int(X_val.shape[0]),
        "n_test_sequences": int(X_test.shape[0]),
        "seq_len": int(X_train.shape[1]),
        "window_size": int(X_train.shape[2]),
        "input_features_signal": int(X_train.shape[3]),
        "input_features_stats_window": int(X_train_stats_seq.shape[2]),
        "n_classes": int(len(classes)),
        "batch_size": int(BATCH_SIZE),
        "epochs": int(EPOCHS),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "best_val_loss": float(min(history.history["val_loss"])),
        "train_time_sec": float(train_elapsed),
        "predict_time_sec": float(predict_elapsed),
        "total_time_sec": float(total_elapsed),
        "stats_feature_names": "|".join(stats_feature_names),
    }
    pd.Series(summary).to_csv(OUTPUT_DIR / "run_summary.csv")

    print("\nArquivos salvos em:", OUTPUT_DIR)
    print("- cnn_gap_v6.keras")
    print("- classification_report_windows.csv")
    print("- classification_report_group.csv")
    print("- confusion_matrix_windows.npy")
    print("- confusion_matrix_group.npy")
    print("- confusion_matrix.png")
    print("- confusion_matrix_normalized.png")
    print("- loss_curve.png")
    print("- accuracy_curve.png")
    print("- run_summary.csv")
    print("- preprocessed_dataset.npz")

    print(f"\nTempo total de execução: {total_elapsed:.2f} s ({total_elapsed / 60:.2f} min)")


if __name__ == "__main__":
    main()