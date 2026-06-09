from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.cv import stratified_group_kfold_split
from src.dataset_builder import (
    build_dataset_from_catalog,
    pack_windows_and_stats_into_sequences,
)
from train_cnn_lstm_v7 import (
    load_filtered_dataset,
    filter_df_by_labels,
    split_train_val_by_group,
)

# =========================================================
# CONFIG GERAL
# =========================================================
LABEL_COL = "falha_base"
GROUP_COL = "grupo_condicao"

WINDOW_SIZE = 1024
OVERLAP = 0.5
DROP_LAST = True
NROWS = None

SEQ_LEN = 10
TRAIN_SEQUENCE_STRIDE = 5
EVAL_SEQUENCE_STRIDE = 5
DROP_INCOMPLETE_SEQUENCES = True

OUTER_FOLD_INDEX = 0
OUTER_N_SPLITS = 2
OUTER_RANDOM_STATE = 42

# ---------------------------------------------------------
# PERFIS
# ---------------------------------------------------------
ACTIVE_PROFILE = "no_bearing_classic"
KEEP_HEALTHY_NOISE_SEPARATE = False

# balanceamento leve no cache
ENABLE_BALANCE_BY_CLASS = True
MAX_SEQUENCES_PER_CLASS_TRAIN = 12000
MAX_SEQUENCES_PER_CLASS_VAL = 3000
MAX_SEQUENCES_PER_CLASS_TEST = 6000

# limite de grupos por classe (opcional)
ENABLE_GROUP_CAP_BY_CLASS = False
MAX_GROUPS_PER_CLASS = 999999

OUTPUT_BASE_DIR = Path("outputs")
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_CONFIGS: dict[str, dict] = {
    "general": {
        "target_labels": [
            "bearing bpfo",
            "bearing bpfi",
            "bearing pump",
            "healthy",
            "healthy noise",
            "impeller",
            "soft foot",
            "bearing bsf",
            "bearing contaminated",
            "broken rotor bar",
            "loose foot pump",
            "loose foot motor",
        ],
        "output_dir": OUTPUT_BASE_DIR / "cnn_cache_motor2_general_v2",
        "cache_name": "cnn_cache_general_w1024_seq1_st5_fold0.npz",
    },
    "no_bearing_classic": {
        "target_labels": [
            "bearing pump",
            "healthy",
            "healthy noise",
            "impeller",
            "soft foot",
            "broken rotor bar",
            "loose foot pump",
            "loose foot motor",
        ],
       "output_dir": OUTPUT_BASE_DIR / "cnn_cache_motor2_no_bpfi_bpfo_seq10_v1",
       "cache_name": "cnn_cache_no_bpfi_bpfo_w1024_seq10_st5_fold0.npz",
    },
}

# =========================================================
# HELPERS
# =========================================================
def normalize_fault_labels(
    df: pd.DataFrame,
    label_col: str,
    keep_healthy_noise_separate: bool,
) -> pd.DataFrame:
    df = df.copy()

    # remove sufixos numéricos: "bearing bpfi 1" -> "bearing bpfi"
    df[label_col] = df["falha"].str.replace(r"\s+\d+$", "", regex=True)

    if not keep_healthy_noise_separate:
        df[label_col] = df[label_col].replace({
            "healthy noise": "healthy",
        })

    return df


def cap_groups_per_class(
    df: pd.DataFrame,
    label_col: str,
    group_col: str,
    max_groups_per_class: int,
) -> pd.DataFrame:
    if max_groups_per_class <= 0:
        return df

    parts = []
    for label, sub in df.groupby(label_col):
        groups = sorted(sub[group_col].astype(str).unique())
        selected_groups = groups[:max_groups_per_class]
        parts.append(sub[sub[group_col].astype(str).isin(selected_groups)])

    out = pd.concat(parts, axis=0).copy()
    return out


def build_sequence_split(
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
        stats_include_optional=False,
        stats_mode="v0_plus_spectral",
        include_band_energy=False,
    )

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

    return X_seq, X_stats_seq, y_seq_raw, meta_seq, stats_feature_names


def balance_sequences_by_class(
    X_seq: np.ndarray,
    X_stats_seq: np.ndarray,
    y_seq_raw: np.ndarray,
    meta_seq: pd.DataFrame,
    max_per_class: int,
    seed: int,
):
    if max_per_class is None or max_per_class <= 0:
        return X_seq, X_stats_seq, y_seq_raw, meta_seq

    rng = np.random.default_rng(seed)
    y_seq_raw = np.array(y_seq_raw)

    keep_indices: list[np.ndarray] = []

    for label in sorted(np.unique(y_seq_raw)):
        idx = np.where(y_seq_raw == label)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
            idx = np.sort(idx)
        keep_indices.append(idx)

    keep_idx = np.concatenate(keep_indices)
    keep_idx = np.sort(keep_idx)

    X_seq = X_seq[keep_idx]
    X_stats_seq = X_stats_seq[keep_idx]
    y_seq_raw = y_seq_raw[keep_idx]
    meta_seq = meta_seq.iloc[keep_idx].reset_index(drop=True)

    return X_seq, X_stats_seq, y_seq_raw, meta_seq


def print_df_summary(df: pd.DataFrame, name: str):
    print(f"\n[{name}] shape: {df.shape}")
    print(f"[{name}] grupos: {df[GROUP_COL].nunique()}")
    print(f"[{name}] classes: {df[LABEL_COL].nunique()}")
    print(f"[{name}] contagem por classe:")
    print(df[LABEL_COL].value_counts().sort_index())


def print_seq_summary(y_seq_raw: np.ndarray, name: str):
    y_seq_raw = np.array(y_seq_raw)
    print(f"\n[{name}] sequências: {len(y_seq_raw)}")
    print(f"[{name}] classes nas sequências: {len(np.unique(y_seq_raw))}")
    print(f"[{name}] contagem por classe nas sequências:")
    print(pd.Series(y_seq_raw).value_counts().sort_index())

#MAX_SAMPLES_PER_CLASS = 4000

#import numpy as np
#import pandas as pd

#y_series = pd.Series(y_train)

#indices_keep = []

#for classe in y_series.unique():
#    idx = np.where(y_train == classe)[0]
#    
#    if len(idx) > MAX_SAMPLES_PER_CLASS:
#        idx = np.random.choice(idx, MAX_SAMPLES_PER_CLASS, replace=False)
#    
#    indices_keep.append(idx)

#indices_keep = np.concatenate(indices_keep)

# embaralha pra não ficar enviesado
#np.random.shuffle(indices_keep)

# aplica o corte
#X_train = X_train[indices_keep]
#y_train = y_train[indices_keep]

# =========================================================
# MAIN
# =========================================================
def main():
    start = time.time()

    if ACTIVE_PROFILE not in PROFILE_CONFIGS:
        raise ValueError(
            f"ACTIVE_PROFILE inválido: {ACTIVE_PROFILE}. "
            f"Opções: {list(PROFILE_CONFIGS.keys())}"
        )

    cfg = PROFILE_CONFIGS[ACTIVE_PROFILE]
    target_labels = cfg["target_labels"]
    output_dir: Path = cfg["output_dir"]
    cache_path: Path = output_dir / cfg["cache_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Perfil ativo: {ACTIVE_PROFILE}")
    print(f"Saída: {cache_path}")
    print(f"KEEP_HEALTHY_NOISE_SEPARATE = {KEEP_HEALTHY_NOISE_SEPARATE}")

    print("\nCarregando dataset filtrado...")
    df = load_filtered_dataset()

    path_cols = [c for c in df.columns if c.startswith("path_ch")]

    for col in path_cols:
        df[col] = df[col].astype(str).str.replace(
            r"F:\TCC\Dataset\NLM-TCC\pump_fault_tcc\CNN\data_raw",
            r"F:\TCC\Dataset\NLM-TCC\pump_fault_tcc\data_raw",
            regex=False,
        )

    print("[PATH FIX] CNN\\data_raw -> data_raw")

    df = normalize_fault_labels(
        df=df,
        label_col=LABEL_COL,
        keep_healthy_noise_separate=KEEP_HEALTHY_NOISE_SEPARATE,
    )

    df = filter_df_by_labels(
        df,
        labels=target_labels,
        label_col=LABEL_COL,
    )

    print("\n[DEBUG LABELS ANTES DO FILTRO]")
    print(sorted(df[LABEL_COL].astype(str).unique()))

    print("\n[DEBUG FOOT LABELS ANTES DO FILTRO]")
    print(
        df[df[LABEL_COL].astype(str).str.contains("foot", case=False, na=False)]
        [LABEL_COL]
        .value_counts()
        .sort_index()
    )

    if ENABLE_GROUP_CAP_BY_CLASS:
        df = cap_groups_per_class(
            df=df,
            label_col=LABEL_COL,
            group_col=GROUP_COL,
            max_groups_per_class=MAX_GROUPS_PER_CLASS,
        )

    print_df_summary(df, "DATASET_FILTRADO")

    outer_splits = stratified_group_kfold_split(
        df=df,
        group_col=GROUP_COL,
        label_col=LABEL_COL,
        n_splits=OUTER_N_SPLITS,
        random_state=OUTER_RANDOM_STATE,
    )
    train_df, test_df = outer_splits[OUTER_FOLD_INDEX]

    train_inner_df, val_df = split_train_val_by_group(
        train_df=train_df,
        label_col=LABEL_COL,
        group_col=GROUP_COL,
        val_size=0.1,
        random_state=123,
    )

    print_df_summary(train_inner_df, "TRAIN")
    print_df_summary(val_df, "VAL")
    print_df_summary(test_df, "TEST")

    # -------------------------
    # BUILD TREINO
    # -------------------------
    print("\n[Build] treino...")
    X_train_raw, X_train_stats_raw, y_train_raw, meta_train, stats_feature_names = build_sequence_split(
        df_subset=train_inner_df,
        label_col=LABEL_COL,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        drop_last=DROP_LAST,
        nrows=NROWS,
        seq_len=SEQ_LEN,
        sequence_stride=TRAIN_SEQUENCE_STRIDE,
        drop_incomplete_sequences=DROP_INCOMPLETE_SEQUENCES,
        grouping_col=GROUP_COL,
    )

    if ENABLE_BALANCE_BY_CLASS:
        X_train_raw, X_train_stats_raw, y_train_raw, meta_train = balance_sequences_by_class(
            X_seq=X_train_raw,
            X_stats_seq=X_train_stats_raw,
            y_seq_raw=y_train_raw,
            meta_seq=meta_train,
            max_per_class=MAX_SEQUENCES_PER_CLASS_TRAIN,
            seed=42,
        )

    print_seq_summary(y_train_raw, "TRAIN_SEQ")

    # -------------------------
    # BUILD VAL
    # -------------------------
    print("\n[Build] validação...")
    X_val_raw, X_val_stats_raw, y_val_raw, meta_val, _ = build_sequence_split(
        df_subset=val_df,
        label_col=LABEL_COL,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        drop_last=DROP_LAST,
        nrows=NROWS,
        seq_len=SEQ_LEN,
        sequence_stride=EVAL_SEQUENCE_STRIDE,
        drop_incomplete_sequences=DROP_INCOMPLETE_SEQUENCES,
        grouping_col=GROUP_COL,
    )

    if ENABLE_BALANCE_BY_CLASS:
        X_val_raw, X_val_stats_raw, y_val_raw, meta_val = balance_sequences_by_class(
            X_seq=X_val_raw,
            X_stats_seq=X_val_stats_raw,
            y_seq_raw=y_val_raw,
            meta_seq=meta_val,
            max_per_class=MAX_SEQUENCES_PER_CLASS_VAL,
            seed=43,
        )

    print_seq_summary(y_val_raw, "VAL_SEQ")

    # -------------------------
    # BUILD TESTE
    # -------------------------
    print("\n[Build] teste...")
    X_test_raw, X_test_stats_raw, y_test_raw, meta_test, _ = build_sequence_split(
        df_subset=test_df,
        label_col=LABEL_COL,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        drop_last=DROP_LAST,
        nrows=NROWS,
        seq_len=SEQ_LEN,
        sequence_stride=EVAL_SEQUENCE_STRIDE,
        drop_incomplete_sequences=DROP_INCOMPLETE_SEQUENCES,
        grouping_col=GROUP_COL,
    )

    if ENABLE_BALANCE_BY_CLASS:
        X_test_raw, X_test_stats_raw, y_test_raw, meta_test = balance_sequences_by_class(
            X_seq=X_test_raw,
            X_stats_seq=X_test_stats_raw,
            y_seq_raw=y_test_raw,
            meta_seq=meta_test,
            max_per_class=MAX_SEQUENCES_PER_CLASS_TEST,
            seed=44,
        )

    print_seq_summary(y_test_raw, "TEST_SEQ")

    # -------------------------
    # SHAPES
    # -------------------------
    print("\nShapes:")
    print("X_train_raw:", X_train_raw.shape)
    print("X_val_raw:", X_val_raw.shape)
    print("X_test_raw:", X_test_raw.shape)
    print("X_train_stats_raw:", X_train_stats_raw.shape)
    print("X_val_stats_raw:", X_val_stats_raw.shape)
    print("X_test_stats_raw:", X_test_stats_raw.shape)
    print("y_train_raw:", np.array(y_train_raw).shape)
    print("y_val_raw:", np.array(y_val_raw).shape)
    print("y_test_raw:", np.array(y_test_raw).shape)

    np.savez_compressed(
        cache_path,
        X_train_raw=X_train_raw.astype(np.float32),
        X_val_raw=X_val_raw.astype(np.float32),
        X_test_raw=X_test_raw.astype(np.float32),
        X_train_stats_raw=X_train_stats_raw.astype(np.float32),
        X_val_stats_raw=X_val_stats_raw.astype(np.float32),
        X_test_stats_raw=X_test_stats_raw.astype(np.float32),
        y_train_raw=np.array(y_train_raw),
        y_val_raw=np.array(y_val_raw),
        y_test_raw=np.array(y_test_raw),
        meta_train_group=np.array(meta_train[GROUP_COL].astype(str)),
        meta_val_group=np.array(meta_val[GROUP_COL].astype(str)),
        meta_test_group=np.array(meta_test[GROUP_COL].astype(str)),
        stats_feature_names=np.array(stats_feature_names, dtype=object),
    )

    summary_rows = [
        {
            "split": "train",
            "n_sequences": int(len(y_train_raw)),
            "sequence_stride": TRAIN_SEQUENCE_STRIDE,
            "window_size": WINDOW_SIZE,
            "seq_len": SEQ_LEN,
        },
        {
            "split": "val",
            "n_sequences": int(len(y_val_raw)),
            "sequence_stride": EVAL_SEQUENCE_STRIDE,
            "window_size": WINDOW_SIZE,
            "seq_len": SEQ_LEN,
        },
        {
            "split": "test",
            "n_sequences": int(len(y_test_raw)),
            "sequence_stride": EVAL_SEQUENCE_STRIDE,
            "window_size": WINDOW_SIZE,
            "seq_len": SEQ_LEN,
        },
    ]
    pd.DataFrame(summary_rows).to_csv(output_dir / "cache_summary.csv", index=False)

    train_counts = pd.Series(np.array(y_train_raw)).value_counts().sort_index().rename("train")
    val_counts = pd.Series(np.array(y_val_raw)).value_counts().sort_index().rename("val")
    test_counts = pd.Series(np.array(y_test_raw)).value_counts().sort_index().rename("test")
    class_dist = pd.concat([train_counts, val_counts, test_counts], axis=1).fillna(0).astype(int)
    class_dist.to_csv(output_dir / "class_distribution_sequences.csv")

    elapsed = time.time() - start
    print(f"\nCache salvo em: {cache_path}")
    print(f"Tempo total: {elapsed:.2f}s ({elapsed / 60:.2f} min)")


if __name__ == "__main__":
    main()