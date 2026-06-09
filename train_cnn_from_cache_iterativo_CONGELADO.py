from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    TimeDistributed,
    Conv1D,
    BatchNormalization,
    ReLU,
    MaxPooling1D,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from train_cnn_lstm_v7 import (
    encode_with_train_only,
    fit_transform_sequence_train_val_test,
)


# =========================================================
# CONFIG GERAL
# =========================================================
CACHE_PATH = Path(
    r"F:\TCC\Dataset\NLM-TCC\pump_fault_tcc\outputs\cnn_cache_motor2_no_bearing_classic_v1\cnn_cache_no_bearing_classic_w1024_seq1_st5_fold0.npz"
)
BASE_OUTPUT = Path(
    r"F:\TCC\Dataset\NLM-TCC\pump_fault_tcc\outputs\resultado_iterativo"
)
BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

RUN_NAME = "cnn_hierarchical_simple"

# MODO HIERÁRQUICO: True = 2 níveis, False = flat normal
HIERARCHICAL_MODE = True

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 2e-5

DROP_UNSEEN_LABELS = False  # Desativado para evitar erro de memória
SAVE_CM_CSV = True


# =========================================================
# HELPERS
# =========================================================
def create_run_dir(config_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_OUTPUT / f"{ts}_{config_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def add_energy_feature(X: np.ndarray) -> np.ndarray:
    """
    Adiciona 1 canal de energia global normalizada por amostra.
    Entrada:  (n, 1024, c)
    Saída:    (n, 1024, c+1)
    """
    X_energy = np.sum(X**2, axis=2, keepdims=True)  # (n, 1024, 1)
    X_energy = X_energy / (np.max(X_energy, axis=1, keepdims=True) + 1e-8)
    return np.concatenate([X, X_energy], axis=2)

def add_band_energy_feature(X_base, fs=25600):
    """
    X_base: (n_amostras, 1024, n_canais)
    Retorna 3 canais extras com energia relativa por banda física.
    """
    eps = 1e-8
    n_samples, win_size, n_channels = X_base.shape

    freqs = np.fft.rfftfreq(win_size, d=1.0 / fs)
    X_fft_power = np.abs(np.fft.rfft(X_base, axis=1)) ** 2
    # shape: (n_amostras, n_freq_bins, n_canais)

    bands = [
        (0, 1000),
        (1000, 3000),
        (3000, 6000),
        (6000, 9000),
        (9000, 11000),  # 🔥 principal
        (11000, 13000),
    ]

    total_power = X_fft_power.sum(axis=(1, 2), keepdims=True) + eps
    band_features = []

    for f_low, f_high in bands:
        mask = (freqs >= f_low) & (freqs < f_high)
        band_power = X_fft_power[:, mask, :].sum(axis=(1, 2), keepdims=True) / total_power
        band_power = np.repeat(band_power, win_size, axis=1)   # (n, 1024, 1)
        band_features.append(band_power)

    return np.concatenate(band_features, axis=2).astype(np.float32)

    # cópia do sinal base escalado para bandas físicas
    X_train_base = X_train.copy()
    X_val_base = X_val.copy()
    X_test_base = X_test.copy()

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
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

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

    if SAVE_CM_CSV:
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(
            output_dir / "confusion_matrix.csv"
        )
        pd.DataFrame(cm_norm, index=classes, columns=classes).to_csv(
            output_dir / "confusion_matrix_normalized.csv"
        )


def build_cnn_simple(seq_len: int, window_size: int, n_features: int, n_classes: int) -> tf.keras.Model:
    """
    CNN 4D com Dual Pooling (Max + Avg) para melhor separabilidade.
    """
    inputs = Input(shape=(seq_len, window_size, n_features), name="sequence_input")

    x = TimeDistributed(Conv1D(32, 9, padding="same", use_bias=False))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)
    x = TimeDistributed(MaxPooling1D(2))(x)
    x = TimeDistributed(Dropout(0.20))(x)

    x = TimeDistributed(Conv1D(64, 7, padding="same", use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)
    x = TimeDistributed(MaxPooling1D(2))(x)
    x = TimeDistributed(Dropout(0.25))(x)

    x = TimeDistributed(Conv1D(128, 5, padding="same", use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)
    x = TimeDistributed(MaxPooling1D(2))(x)
    x = TimeDistributed(Dropout(0.30))(x)

    x = TimeDistributed(Conv1D(128, 3, padding="same", use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(ReLU())(x)

    # DUAL POOLING: captura picos (max) e média (avg)
    x_max = TimeDistributed(GlobalMaxPooling1D())(x)
    x_avg = TimeDistributed(GlobalAveragePooling1D())(x)
    x = Concatenate()([x_max, x_avg])  # (batch, seq_len, 256)

    # Agrega as sequências
    x = GlobalAveragePooling1D()(x)

    # Classificador um pouco maior para lidar com 256 features
    x = Dense(192, activation='relu')(x)
    x = Dropout(0.40)(x)
    outputs = Dense(n_classes, activation="softmax", name="classifier")(x)

    model = Model(inputs, outputs, name="cnn_sequence_dual_pooling")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def filter_unseen_labels(
    X_train_raw: np.ndarray,
    X_val_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_train_raw: np.ndarray,
    y_val_raw: np.ndarray,
    y_test_raw: np.ndarray,
):
    train_labels = set(np.unique(y_train_raw))
    unseen_val = sorted(set(np.unique(y_val_raw)) - train_labels)
    unseen_test = sorted(set(np.unique(y_test_raw)) - train_labels)

    print("\n[CHECK] labels no treino:", sorted(train_labels))
    print("[CHECK] labels fora do treino na val:", unseen_val)
    print("[CHECK] labels fora do treino no test:", unseen_test)

    if not DROP_UNSEEN_LABELS:
        return (
            X_train_raw, X_val_raw, X_test_raw,
            y_train_raw, y_val_raw, y_test_raw,
            unseen_val, unseen_test
        )

    keep_val = np.isin(y_val_raw, list(train_labels))
    keep_test = np.isin(y_test_raw, list(train_labels))

    dropped_val = int((~keep_val).sum())
    dropped_test = int((~keep_test).sum())

    if dropped_val > 0 or dropped_test > 0:
        print("\n[FILTER] Removendo amostras com labels fora do treino.")
        print(f"[FILTER] val removidas: {dropped_val}")
        print(f"[FILTER] test removidas: {dropped_test}")

    # Filtrar de forma mais eficiente (sem copiar arrays grandes)
    if dropped_val > 0:
        indices_val = np.where(keep_val)[0]
        X_val_raw = X_val_raw[indices_val]
        y_val_raw = y_val_raw[indices_val]
        del indices_val  # liberar memória

    if dropped_test > 0:
        indices_test = np.where(keep_test)[0]
        X_test_raw = X_test_raw[indices_test]
        y_test_raw = y_test_raw[indices_test]
        del indices_test  # liberar memória

    return (
        X_train_raw, X_val_raw, X_test_raw,
        y_train_raw, y_val_raw, y_test_raw,
        unseen_val, unseen_test
    )


def print_saved_files(run_dir: Path):
    expected = [
        "classification_report.csv",
        "confusion_matrix.npy",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "loss_curve.png",
        "accuracy_curve.png",
        "model.keras",
        "run_summary.csv",
    ]
    if SAVE_CM_CSV:
        expected.extend([
            "confusion_matrix.csv",
            "confusion_matrix_normalized.csv",
        ])

    print("\nArquivos gerados:")
    for fname in expected:
        fpath = run_dir / fname
        status = "OK" if fpath.exists() else "FALTOU"
        print(f" - {status} -> {fpath}")


def train_hierarchical(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    classes, run_dir
):
    """
    Treinamento hierárquico em 2 níveis:
    - Nível 1: impeller vs não-impeller (balanceamento agressivo 1:1)
    - Nível 2: 5 classes restantes
    """
    print("\n" + "="*60)
    print("MODO HIERÁRQUICO ATIVADO")
    print("="*60)
    
    # Identificar índice do impeller
    impeller_idx = classes.index("impeller")
    print(f"\n[LEVEL 1] impeller_idx = {impeller_idx}")
    
    # =========================================================
    # NÍVEL 1: CLASSIFICADOR BINÁRIO
    # =========================================================
    print("\n" + "-"*60)
    print("NÍVEL 1: impeller vs não-impeller")
    print("-"*60)
    
    # Criar labels binários (0=não-impeller, 1=impeller)
    y_train_binary = (y_train == impeller_idx).astype(int)
    y_val_binary = (y_val == impeller_idx).astype(int)
    y_test_binary = (y_test == impeller_idx).astype(int)
    
    print(f"\n[LEVEL 1] Distribuição original no treino:")
    print(f"  não-impeller: {(y_train_binary == 0).sum()}")
    print(f"  impeller: {(y_train_binary == 1).sum()}")
    
    # BALANCEAMENTO AGRESSIVO: undersample impeller para 1:1
    idx_non_impeller = np.where(y_train_binary == 0)[0]
    idx_impeller = np.where(y_train_binary == 1)[0]
    
    n_non_impeller = len(idx_non_impeller)
    n_impeller = len(idx_impeller)
    
    # Undersample impeller para igualar não-impeller
    if n_impeller > n_non_impeller:
        idx_impeller = np.random.choice(idx_impeller, n_non_impeller, replace=False)
    else:
        # Se impeller for menor, undersample não-impeller
        idx_non_impeller = np.random.choice(idx_non_impeller, n_impeller, replace=False)
    
    # Combinar índices balanceados
    idx_balanced = np.concatenate([idx_non_impeller, idx_impeller])
    np.random.shuffle(idx_balanced)
    
    X_train_level1 = X_train[idx_balanced]
    y_train_level1 = y_train_binary[idx_balanced]
    
    print(f"\n[LEVEL 1] Após balanceamento 1:1:")
    print(f"  não-impeller: {(y_train_level1 == 0).sum()}")
    print(f"  impeller: {(y_train_level1 == 1).sum()}")
    
    # Verificar validação
    print(f"\n[LEVEL 1] Validação:")
    print(f"  não-impeller: {(y_val_binary == 0).sum()}")
    print(f"  impeller: {(y_val_binary == 1).sum()}")
    
    # Class weights para nível 1
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights_level1_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_level1),
        y=y_train_level1,
    )
    class_weights_level1 = {
        int(cls): float(weight)
        for cls, weight in zip(np.unique(y_train_level1), class_weights_level1_array)
    }
    
    print(f"\n[LEVEL 1] Class weights:")
    print(f"  não-impeller: {class_weights_level1.get(0, 1.0):.3f}")
    print(f"  impeller: {class_weights_level1.get(1, 1.0):.3f}")
    
    # Construir modelo nível 1
    model_level1 = build_cnn_simple(
        seq_len=X_train_level1.shape[1],
        window_size=X_train_level1.shape[2],
        n_features=X_train_level1.shape[3],
        n_classes=2,
    )
    
    print("\n[LEVEL 1] Arquitetura:")
    model_level1.summary()
    
    # Callbacks para nível 1
    callbacks_level1 = [
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    
    print("\n[LEVEL 1] Treinando...")
    history_level1 = model_level1.fit(
        X_train_level1,
        y_train_level1,
        validation_data=(X_val, y_val_binary),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights_level1,
        callbacks=callbacks_level1,
        verbose=1,
        shuffle=True,
    )
    
    # Salvar modelo nível 1
    model_level1.save(run_dir / "model_level1.keras")
    
    # Predição nível 1 em batches
    print("\n[LEVEL 1] Predizendo no teste...")
    y_pred_level1_proba = []
    n_batches = int(np.ceil(len(X_test) / BATCH_SIZE))
    
    for i in range(n_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(X_test))
        batch_pred = model_level1.predict(X_test[start_idx:end_idx], batch_size=BATCH_SIZE, verbose=0)
        y_pred_level1_proba.append(batch_pred)
    
    y_pred_level1_proba = np.vstack(y_pred_level1_proba)
    y_pred_level1 = np.argmax(y_pred_level1_proba, axis=1)
    
    # Avaliar nível 1
    print("\n[LEVEL 1] Resultados no teste:")
    print(classification_report(
        y_test_binary,
        y_pred_level1,
        target_names=["não-impeller", "impeller"],
        zero_division=0,
    ))
    
    cm_level1 = confusion_matrix(y_test_binary, y_pred_level1)
    print("\n[LEVEL 1] Matriz de confusão:")
    print(cm_level1)
    
    # =========================================================
    # NÍVEL 2: CLASSIFICADOR MULTI-CLASSE (5 classes)
    # =========================================================
    print("\n" + "-"*60)
    print("NÍVEL 2: 5 classes (sem impeller)")
    print("-"*60)
    
    # Filtrar dados não-impeller no treino
    mask_train_non_impeller = (y_train != impeller_idx)
    X_train_level2 = X_train[mask_train_non_impeller]
    y_train_level2_original = y_train[mask_train_non_impeller]
    
    # Remapear labels para 0-4 (sem impeller)
    classes_level2 = [c for c in classes if c != "impeller"]
    
    # Criar mapeamento old_idx -> new_idx
    old_to_new = {}
    for new_idx, cls in enumerate(classes_level2):
        old_idx = classes.index(cls)
        old_to_new[old_idx] = new_idx
    
    y_train_level2 = np.array([old_to_new[y] for y in y_train_level2_original])
    
    print(f"\n[LEVEL 2] Classes: {classes_level2}")
    print(f"\n[LEVEL 2] Distribuição no treino:")
    for cls_idx, cls_name in enumerate(classes_level2):
        print(f"  {cls_name}: {(y_train_level2 == cls_idx).sum()}")
    
    # Filtrar validação não-impeller (se houver)
    mask_val_non_impeller = (y_val != impeller_idx)
    
    if mask_val_non_impeller.sum() > 0:
        X_val_level2 = X_val[mask_val_non_impeller]
        y_val_level2_original = y_val[mask_val_non_impeller]
        y_val_level2 = np.array([old_to_new[y] for y in y_val_level2_original])
        
        print(f"\n[LEVEL 2] Validação disponível: {len(y_val_level2)} amostras")
        validation_data_level2 = (X_val_level2, y_val_level2)
        callbacks_level2 = [
            EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]
    else:
        print(f"\n[LEVEL 2] SEM validação - usando apenas train loss")
        validation_data_level2 = None
        callbacks_level2 = [
            EarlyStopping(
                monitor="loss",
                patience=20,
                restore_best_weights=True,
                verbose=1,
            ),
        ]
    
    # Class weights para nível 2
    class_weights_level2_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_level2),
        y=y_train_level2,
    )
    class_weights_level2 = {
        int(cls): float(weight)
        for cls, weight in zip(np.unique(y_train_level2), class_weights_level2_array)
    }
    
    print(f"\n[LEVEL 2] Class weights:")
    for cls_idx, cls_name in enumerate(classes_level2):
        weight = class_weights_level2.get(cls_idx, 1.0)
        print(f"  {cls_name}: {weight:.3f}")
    
    # Construir modelo nível 2
    model_level2 = build_cnn_simple(
        seq_len=X_train_level2.shape[1],
        window_size=X_train_level2.shape[2],
        n_features=X_train_level2.shape[3],
        n_classes=len(classes_level2),
    )
    
    print("\n[LEVEL 2] Arquitetura:")
    model_level2.summary()
    
    print("\n[LEVEL 2] Treinando...")
    history_level2 = model_level2.fit(
        X_train_level2,
        y_train_level2,
        validation_data=validation_data_level2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights_level2,
        callbacks=callbacks_level2,
        verbose=1,
        shuffle=True,
    )
    
    # Salvar modelo nível 2
    model_level2.save(run_dir / "model_level2.keras")
    
    # =========================================================
    # PREDIÇÃO HIERÁRQUICA FINAL
    # =========================================================
    print("\n" + "-"*60)
    print("PREDIÇÃO HIERÁRQUICA FINAL")
    print("-"*60)
    
    # Inicializar predição final
    y_pred_final = np.zeros(len(y_test), dtype=int)
    
    # Amostras classificadas como impeller no nível 1
    mask_pred_impeller = (y_pred_level1 == 1)
    y_pred_final[mask_pred_impeller] = impeller_idx
    
    print(f"\n[FINAL] Amostras classificadas como impeller: {mask_pred_impeller.sum()}")
    
    # Amostras classificadas como não-impeller vão para nível 2
    mask_pred_non_impeller = (y_pred_level1 == 0)
    X_test_level2 = X_test[mask_pred_non_impeller]
    
    print(f"[FINAL] Amostras para nível 2: {mask_pred_non_impeller.sum()}")
    
    if mask_pred_non_impeller.sum() > 0:
        # Predição nível 2 em batches
        y_pred_level2_proba = []
        n_batches_level2 = int(np.ceil(len(X_test_level2) / BATCH_SIZE))
        
        for i in range(n_batches_level2):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(X_test_level2))
            batch_pred = model_level2.predict(X_test_level2[start_idx:end_idx], batch_size=BATCH_SIZE, verbose=0)
            y_pred_level2_proba.append(batch_pred)
        
        y_pred_level2_proba = np.vstack(y_pred_level2_proba)
        y_pred_level2 = np.argmax(y_pred_level2_proba, axis=1)
        
        # Remapear de volta para índices originais
        new_to_old = {new_idx: classes.index(cls) for new_idx, cls in enumerate(classes_level2)}
        y_pred_level2_remapped = np.array([new_to_old[y] for y in y_pred_level2])
        
        y_pred_final[mask_pred_non_impeller] = y_pred_level2_remapped
    
    # =========================================================
    # AVALIAÇÃO FINAL
    # =========================================================
    print("\n" + "="*60)
    print("RESULTADO FINAL HIERÁRQUICO")
    print("="*60)
    
    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred_final,
        labels=np.arange(len(classes)),
        target_names=classes,
        zero_division=0,
    ))
    
    cm_final = confusion_matrix(y_test, y_pred_final)
    
    # Salvar resultados
    report_df = pd.DataFrame(
        classification_report(
            y_test,
            y_pred_final,
            labels=np.arange(len(classes)),
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_df.to_csv(run_dir / "classification_report.csv", index=True)
    
    np.save(run_dir / "confusion_matrix.npy", cm_final)
    
    plot_confusion_matrix(cm_final, classes, run_dir)
    
    # Salvar históricos
    (run_dir / "level1").mkdir(exist_ok=True)
    plot_training_history(history_level1, run_dir / "level1")
    
    (run_dir / "level2").mkdir(exist_ok=True)
    plot_training_history(history_level2, run_dir / "level2")
    
    return y_pred_final, cm_final


# =========================================================
# MAIN
# =========================================================
def main():
    global_start = time.time()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    print("GPUs disponíveis:", tf.config.list_physical_devices("GPU"))


    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Cache não encontrado: {CACHE_PATH}")

    run_dir = create_run_dir(RUN_NAME)


    print("\nCarregando cache...")
    data = np.load(CACHE_PATH, allow_pickle=True)

    X_train_raw = data["X_train_raw"]
    X_val_raw = data["X_val_raw"]
    X_test_raw = data["X_test_raw"]

    y_train_raw = data["y_train_raw"]
    y_val_raw = data["y_val_raw"]
    y_test_raw = data["y_test_raw"]

    print("\nShapes do cache:")
    print("X_train_raw:", X_train_raw.shape)
    print("X_val_raw:", X_val_raw.shape)
    print("X_test_raw:", X_test_raw.shape)
    print("y_train_raw:", y_train_raw.shape)
    print("y_val_raw:", y_val_raw.shape)
    print("y_test_raw:", y_test_raw.shape)

    (
        X_train_raw,
        X_val_raw,
        X_test_raw,
        y_train_raw,
        y_val_raw,
        y_test_raw,
        unseen_val,
        unseen_test,
    ) = filter_unseen_labels(
        X_train_raw,
        X_val_raw,
        X_test_raw,
        y_train_raw,
        y_val_raw,
        y_test_raw,
    )

    y_train, y_val, y_test, label_encoder = encode_with_train_only(
        y_train_raw=y_train_raw,
        y_val_raw=y_val_raw,
        y_test_raw=y_test_raw,
    )
    classes = [str(c) for c in label_encoder.classes_]

    print("\nClasses finais no encoder:")
    print(classes)

    # ===== MERGE healthy + healthy noise COM REINDEXAÇÃO =====
    from sklearn.preprocessing import LabelEncoder
    
    classes_original = list(label_encoder.classes_)
    print("\n[MERGE] Classes originais:", classes_original)
    
    if "healthy" in classes_original and "healthy noise" in classes_original:
        idx_healthy = classes_original.index("healthy")
        idx_noise = classes_original.index("healthy noise")
        
        # substitui healthy noise por healthy
        y_train[y_train == idx_noise] = idx_healthy
        y_val[y_val == idx_noise] = idx_healthy
        y_test[y_test == idx_noise] = idx_healthy
        
        # reconstrói classes sem healthy noise
        classes_new = [c for c in classes_original if c != "healthy noise"]
        
        # cria novo encoder
        new_encoder = LabelEncoder()
        new_encoder.classes_ = np.array(classes_new)
        
        # remapeia labels para índices contíguos
        old_to_new = {}
        for new_idx, cls in enumerate(classes_new):
            old_idx = classes_original.index(cls)
            old_to_new[old_idx] = new_idx
        
        y_train = np.array([old_to_new[y] for y in y_train])
        y_val = np.array([old_to_new[y] for y in y_val])
        y_test = np.array([old_to_new[y] for y in y_test])
        
        classes = classes_new
        print(f"[MERGE] healthy noise -> healthy aplicado")
        print(f"[MERGE] Classes finais ({len(classes)}):", classes)
    else:
        classes = classes_original
        print("[MERGE] Merge não necessário")

    # cache shape esperado: (n, seq_len, 1024, 5)
    # SCALING COM MEMMAP para evitar erro de memória
    print("\n[SCALING] Usando memmap para economizar RAM...")
    
    from sklearn.preprocessing import StandardScaler
    import tempfile
    
    # Fit scaler em amostra do treino
    sample_size = min(5000, len(X_train_raw))
    X_train_sample = X_train_raw[:sample_size, 0, :, :]  # (5000, 1024, 5)
    n_samples, win_size, n_channels = X_train_sample.shape
    X_train_2d = X_train_sample.reshape(-1, n_channels)  # (5000*1024, 5)
    
    scaler = StandardScaler()
    scaler.fit(X_train_2d)
    
    print(f"[SCALING] Scaler fitted em {X_train_2d.shape[0]} pontos")
    del X_train_sample, X_train_2d
    
    # Transform usando memmap temporário
    def scale_4d_memmap(X_4d, scaler, batch_size=500):
        n, seq_len, win, ch = X_4d.shape
        
        # Criar memmap temporário
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
        temp_path = temp_file.name
        temp_file.close()
        
        X_scaled = np.memmap(temp_path, dtype='float32', mode='w+', shape=(n, seq_len, win, ch))
        
        n_batches = int(np.ceil(n / batch_size))
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n)
            
            X_batch = X_4d[start_idx:end_idx]
            
            for seq_idx in range(seq_len):
                X_window = X_batch[:, seq_idx, :, :]  # (batch, 1024, 5)
                batch_n = X_window.shape[0]
                X_2d = X_window.reshape(-1, ch)
                X_2d_scaled = scaler.transform(X_2d)
                X_scaled[start_idx:end_idx, seq_idx, :, :] = X_2d_scaled.reshape(batch_n, win, ch)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1}/{n_batches}")
        
        # Converter para array normal
        X_result = np.array(X_scaled, dtype=np.float32)
        del X_scaled
        
        # Remover arquivo temporário
        import os
        try:
            os.unlink(temp_path)
        except:
            pass
        
        return X_result
    
    print("[SCALING] Transformando train...")
    X_train = scale_4d_memmap(X_train_raw, scaler, batch_size=500)
    
    print("[SCALING] Transformando val...")
    X_val = scale_4d_memmap(X_val_raw, scaler, batch_size=500)
    
    print("[SCALING] Transformando test...")
    X_test = scale_4d_memmap(X_test_raw, scaler, batch_size=500)
    
    print("[SCALING] Concluído")

    print("\nShapes após scaling seq:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    
    # ===== LOW-PASS DESATIVADO (PIORAVA RESULTADOS) =====
    # def lowpass_signal(x, cutoff_ratio=0.6):
    #     Xf = np.fft.rfft(x, axis=0)
    #     cutoff = int(Xf.shape[0] * cutoff_ratio)
    #     Xf[cutoff:] = 0
    #     return np.fft.irfft(Xf, n=x.shape[0], axis=0).astype(np.float32)
    # 
    # X_train = np.array([lowpass_signal(w) for w in X_train], dtype=np.float32)
    # X_val = np.array([lowpass_signal(w) for w in X_val], dtype=np.float32)
    # X_test = np.array([lowpass_signal(w) for w in X_test], dtype=np.float32)
    # print("[FILTER] Low-pass aplicado")

    print("\nShapes finais:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    print("y_test:", y_test.shape)

    # ===== BASE PARA BANDAS (ANTES DE QUALQUER FEATURE) =====
   # X_train_base = X_train.copy()
   # X_val_base = X_val.copy()
   # X_test_base = X_test.copy()

    # ===== FEATURE: ENERGIA GLOBAL =====
    #X_train = add_energy_feature(X_train)
    #X_val = add_energy_feature(X_val)
    #X_test = add_energy_feature(X_test)

    
    #print("[FEATURE] Energia adicionada")
    #print("Shape após energia:", X_train.shape)

    # ===== FEATURE: DERIVADA =====
    #X_train_diff = np.diff(X_train, axis=1, prepend=0)
    #X_val_diff = np.diff(X_val, axis=1, prepend=0)
    #X_test_diff = np.diff(X_test, axis=1, prepend=0)

    #X_train = np.concatenate([X_train, X_train_diff], axis=2)
    #X_val = np.concatenate([X_val, X_val_diff], axis=2)
    #X_test = np.concatenate([X_test, X_test_diff], axis=2)

    #print("[FEATURE] Derivada adicionada")
    #print("Shape após derivada:", X_train.shape)

    # ===== FEATURE: BANDAS FÍSICAS =====
#    X_train_band = add_band_energy_feature(X_train_base, fs=25600)
#    X_val_band = add_band_energy_feature(X_val_base, fs=25600)
#    X_test_band = add_band_energy_feature(X_test_base, fs=25600)

#    X_train = np.concatenate([X_train, X_train_band], axis=2)
#    X_val = np.concatenate([X_val, X_val_band], axis=2)
#    X_test = np.concatenate([X_test, X_test_band], axis=2)

    #print("[FEATURE] Bandas físicas adicionadas")
    #print("Shape após bandas:", X_train.shape)

    print("[FEATURE] Features extras desativadas para teste seq_len=10")
    print("[FEATURE] Shape mantido:", X_train.shape)

    # =========================================================
    # DECISÃO: FLAT vs HIERÁRQUICO
    # =========================================================
    if HIERARCHICAL_MODE:
        print("\n" + "="*60)
        print("MODO SELECIONADO: HIERÁRQUICO")
        print("="*60)
        
        train_start = time.time()
        y_pred, cm = train_hierarchical(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            classes, run_dir
        )
        train_elapsed = time.time() - train_start
        
        # Pular para salvamento final
        total_elapsed = time.time() - global_start
        
        summary = {
            "run_name": RUN_NAME,
            "mode": "hierarchical",
            "cache_path": str(CACHE_PATH),
            "drop_unseen_labels": DROP_UNSEEN_LABELS,
            "batch_size": BATCH_SIZE,
            "epochs_requested": EPOCHS,
            "learning_rate_initial": LEARNING_RATE,
            "n_classes_final": len(classes),
            "classes_final": "|".join(classes),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "input_window": int(X_train.shape[1]),
            "input_features": int(X_train.shape[2]),
            "train_time_sec": float(train_elapsed),
            "total_time_sec": float(total_elapsed),
            "unseen_val_labels": "|".join(unseen_val) if len(unseen_val) else "",
            "unseen_test_labels": "|".join(unseen_test) if len(unseen_test) else "",
        }
        pd.Series(summary).to_csv(run_dir / "run_summary.csv")
        
        print(f"\nFim. Tempo total: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
        print_saved_files(run_dir)
        return
    
    # =========================================================
    # MODO FLAT (ORIGINAL)
    # =========================================================
    print("\n" + "="*60)
    print("MODO SELECIONADO: FLAT (ORIGINAL)")
    print("="*60)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True,
            verbose=1,
        ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    ),
    ]

    model = build_cnn_simple(
        seq_len=X_train.shape[1],
        window_size=X_train.shape[2],
        n_features=X_train.shape[3],
        n_classes=len(classes),
    )
    model.summary()

    print("\nTreinando...")
    train_start = time.time()

    MAX_PER_CLASS = 4000

    print("\n[CAP] Balanceando todas as classes")

    new_indices = []

    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]

        if len(cls_idx) > MAX_PER_CLASS:
            cls_idx = np.random.choice(cls_idx, MAX_PER_CLASS, replace=False)

        new_indices.append(cls_idx)

    new_indices = np.concatenate(new_indices)
    np.random.shuffle(new_indices)

    X_train = X_train[new_indices]
    y_train = y_train[new_indices]

    print("\n[CAP] Depois do balanceamento:")
    for cls in np.unique(y_train):
        cls_name = classes[cls] if cls < len(classes) else f"classe_{cls}"
        print(f"{cls_name}: {(y_train == cls).sum()}")

    #TARGET_PER_CLASS = 4000

    #print("\n[OVERSAMPLE] Equalizando classes para", TARGET_PER_CLASS)

    # unique_classes = np.unique(y_train)
    # new_indices = []

    # for cls in unique_classes:
    #     cls_idx = np.where(y_train == cls)[0]
    #     current_count = len(cls_idx)

    #     if current_count > TARGET_PER_CLASS:
    #         # corta
    #         cls_idx = np.random.choice(cls_idx, TARGET_PER_CLASS, replace=False)
    #     else:
    #         # estufa (repete)
    #         extra = np.random.choice(cls_idx, TARGET_PER_CLASS - current_count, replace=True)
    #         cls_idx = np.concatenate([cls_idx, extra])

    #     new_indices.append(cls_idx)

    # # junta tudo
    # new_indices = np.concatenate(new_indices)
    # np.random.shuffle(new_indices)

    # # aplica nos dados
    # X_train = X_train[new_indices]
    # y_train = y_train[new_indices]

#    print("\n[OVERSAMPLE] Depois do balanceamento:")
#    for cls in unique_classes:
#        print(f"classe {cls}: {(y_train == cls).sum()}")

    from sklearn.utils.class_weight import compute_class_weight

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )

    class_weights = {
        int(cls): float(weight)
        for cls, weight in zip(np.unique(y_train), class_weights_array)
    }
    
    print("\n[WEIGHTS] Class weights calculados:")
    for cls_idx, weight in class_weights.items():
        cls_name = classes[cls_idx] if cls_idx < len(classes) else f"classe_{cls_idx}"
        print(f"  {cls_name:25s}: {weight:.3f}")

#    mask = y_train != classes.index("impeller")

#    X_train = X_train[mask]
#    y_train = y_train[mask]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,  # ATIVADO
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )
        
    train_elapsed = time.time() - train_start
    print(f"\nTempo de treino: {train_elapsed:.2f}s ({train_elapsed/60:.2f} min)")

    print("\nPredizendo no teste...")
    predict_start = time.time()
    
    # predição em batches para evitar erro de GPU
    y_pred_proba = []
    n_batches = int(np.ceil(len(X_test) / BATCH_SIZE))
    
    for i in range(n_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(X_test))
        batch_pred = model.predict(X_test[start_idx:end_idx], batch_size=BATCH_SIZE, verbose=0)
        y_pred_proba.append(batch_pred)
        if (i + 1) % 50 == 0:
            print(f"  Batch {i+1}/{n_batches}")
    
    y_pred_proba = np.vstack(y_pred_proba)
    predict_elapsed = time.time() - predict_start
    print(f"Tempo de predição: {predict_elapsed:.2f}s ({predict_elapsed/60:.2f} min)")

    y_pred = np.argmax(y_pred_proba, axis=1)

    # =====================================================
    # SALVAMENTOS
    # =====================================================
    report_path = run_dir / "classification_report.csv"
    cm_npy_path = run_dir / "confusion_matrix.npy"
    model_path = run_dir / "model.keras"
    summary_path = run_dir / "run_summary.csv"

    report_df = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(classes)),
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_df.to_csv(report_path, index=True)

    cm = confusion_matrix(y_test, y_pred)
    np.save(cm_npy_path, cm)

    plot_training_history(history, run_dir)
    plot_confusion_matrix(cm, classes, run_dir)

    model.save(model_path)

    total_elapsed = time.time() - global_start

    summary = {
        "run_name": RUN_NAME,
        "cache_path": str(CACHE_PATH),
        "drop_unseen_labels": DROP_UNSEEN_LABELS,
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "epochs_executed": len(history.history["loss"]),
        "learning_rate_initial": LEARNING_RATE,
        "n_classes_final": len(classes),
        "classes_final": "|".join(classes),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "input_window": int(X_train.shape[1]),
        "input_features": int(X_train.shape[2]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "best_val_loss": float(min(history.history["val_loss"])),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "train_time_sec": float(train_elapsed),
        "predict_time_sec": float(predict_elapsed),
        "total_time_sec": float(total_elapsed),
        "unseen_val_labels": "|".join(unseen_val) if len(unseen_val) else "",
        "unseen_test_labels": "|".join(unseen_test) if len(unseen_test) else "",
    }
    pd.Series(summary).to_csv(summary_path)

    print("\nClassification report:")
    print(
        classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        zero_division=0,
        )
    )


    print(f"\nFim. Tempo total: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print("Arquivos salvos na pasta com sucesso")


if __name__ == "__main__":
    main()