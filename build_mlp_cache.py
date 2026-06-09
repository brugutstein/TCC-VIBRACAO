from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold, train_test_split


# =========================================================
# BUILD MLP CACHE — VIBRATION ONLY
# =========================================================

PROJECT_ROOT = Path(r"F:\TCC\Dataset\NLM-TCC\pump_fault_tcc")
CATALOG_PATH = PROJECT_ROOT / "outputs" / "catalog_samples.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "mlp_dataset"
    / "mlp_w1024_ov50_motor2_vibration_7cls_psd_bands_v2"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# CONFIG
# =========================
WINDOW_SIZE = 1024
OVERLAP = 0.5
STEP = int(WINDOW_SIZE * (1.0 - OVERLAP))
DROP_LAST = True

FS = 25600

OUTER_N_SPLITS = 3
OUTER_FOLD_INDEX = 0
OUTER_RANDOM_STATE = 42

VAL_SIZE = 0.2
VAL_RANDOM_STATE = 123

MAX_HEALTHY_TRAIN_WINDOWS = 4000
RANDOM_SEED = 42

TARGET_LABELS = [
    "bearing pump",
    "broken rotor bar",
    "healthy",
    "impeller",
    "soft foot",
    "loose foot motor",
    "loose foot pump",
]

EXCLUDED_LABELS = [
    "bearing bpfi",
    "bearing bpfo",
]

LABEL_MERGE = {
    "healthy noise": "healthy",
    "healthy 1": "healthy",
    "healthy 2": "healthy",
    "healthy 3": "healthy",
    "soft foot 1": "soft foot",
    "soft foot 2": "soft foot",
}

BANDS = [
    (0, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 7000),
    (7000, 9000),
    (9000, 11000),
    (11000, 13000),
]


# =========================================================
# LABELS
# =========================================================
def normalize_fault_label(label: str) -> str:
    label = str(label).lower().strip()
    label = label.replace("_", " ")
    label = label.replace("-", " ")
    label = re.sub(r"\s+", " ", label)

    if label in LABEL_MERGE:
        return LABEL_MERGE[label]

    label = re.sub(r"\s+\d+$", "", label)

    return LABEL_MERGE.get(label, label)


# =========================================================
# LOAD CATALOG
# =========================================================
def load_catalog() -> pd.DataFrame:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catálogo não encontrado:\n{CATALOG_PATH}\n\n"
            "Rode primeiro: python catalog.py"
        )

    df = pd.read_csv(CATALOG_PATH)

    required_cols = {
        "sensor",
        "equipamento",
        "rpm",
        "falha",
        "signal_id",
        "grupo_condicao",
        "is_sample_valid",
        "n_channels_found",
        "path_ch1",
        "path_ch2",
        "path_ch3",
        "path_ch4",
        "path_ch5",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no catalog_samples.csv: {sorted(missing)}")

    df = df[
        (df["sensor"].astype(str) == "Vibration")
        & (df["equipamento"].astype(str) == "Motor-2")
        & (df["is_sample_valid"] == True)
        & (df["n_channels_found"] == 5)
    ].copy()

    df["falha_base"] = df["falha"].apply(normalize_fault_label)

    df = df[~df["falha_base"].isin(EXCLUDED_LABELS)].copy()
    df = df[df["falha_base"].isin(TARGET_LABELS)].copy()

    if df.empty:
        raise ValueError("Catálogo ficou vazio após filtros do MLP.")

    df["grupo_condicao_mlp"] = (
        df["sensor"].astype(str)
        + "__"
        + df["equipamento"].astype(str)
        + "__"
        + df["rpm"].astype(str)
        + "__"
        + df["falha_base"].astype(str)
        + "__"
        + df["signal_id"].astype(str)
    )

    return df.reset_index(drop=True)


# =========================================================
# SPLIT SEM VAZAMENTO
# =========================================================
def stratified_group_split(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    n_splits: int,
    random_state: int,
):
    group_df = df[[group_col, label_col]].drop_duplicates().reset_index(drop=True)

    min_groups = int(group_df[label_col].value_counts().min())

    if min_groups < n_splits:
        n_splits = min_groups
        print(f"[WARN] OUTER_N_SPLITS ajustado para {n_splits}")

    if n_splits < 2:
        raise ValueError("Poucos grupos por classe para split sem vazamento.")

    X = group_df[group_col].to_numpy()
    y = group_df[label_col].to_numpy()

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits = []

    for train_idx, test_idx in skf.split(X, y):
        train_groups = set(X[train_idx])
        test_groups = set(X[test_idx])

        train_df = df[df[group_col].isin(train_groups)].copy()
        test_df = df[df[group_col].isin(test_groups)].copy()

        splits.append((train_df, test_df))

    return splits


def split_train_val_by_group(
    train_df: pd.DataFrame,
    group_col: str,
    label_col: str,
    val_size: float,
    random_state: int,
):
    group_df = train_df[[group_col, label_col]].drop_duplicates().reset_index(drop=True)

    n_groups = len(group_df)
    n_classes = group_df[label_col].nunique()
    n_val = int(np.ceil(n_groups * val_size))

    label_counts = group_df[label_col].value_counts()
    can_stratify = label_counts.min() >= 2 and n_val >= n_classes

    train_groups, val_groups = train_test_split(
        group_df[group_col],
        test_size=val_size,
        random_state=random_state,
        stratify=group_df[label_col] if can_stratify else None,
    )

    train_inner_df = train_df[train_df[group_col].isin(train_groups)].copy()
    val_df = train_df[train_df[group_col].isin(val_groups)].copy()

    return train_inner_df, val_df


def save_split_audit(train_df, val_df, test_df):
    g_train = set(train_df["grupo_condicao_mlp"].astype(str))
    g_val = set(val_df["grupo_condicao_mlp"].astype(str))
    g_test = set(test_df["grupo_condicao_mlp"].astype(str))

    inter = pd.DataFrame([
        {"intersection": "train_val", "n_common": len(g_train & g_val)},
        {"intersection": "train_test", "n_common": len(g_train & g_test)},
        {"intersection": "val_test", "n_common": len(g_val & g_test)},
    ])

    inter.to_csv(OUTPUT_DIR / "split_intersections.csv", index=False)

    train_df.to_csv(OUTPUT_DIR / "split_train_samples.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "split_val_samples.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "split_test_samples.csv", index=False)

    print("\n[SPLIT INTERSECTIONS]")
    print(inter)

    if inter["n_common"].sum() != 0:
        raise RuntimeError("Vazamento detectado entre splits.")


# =========================================================
# LEITURA 5 CANAIS VIBRAÇÃO
# =========================================================
def resolve_path(path_value: str) -> Path:
    raw = str(path_value).strip()

    candidates = [
        Path(raw),
        PROJECT_ROOT / raw,
        PROJECT_ROOT / raw.replace("\\", "/"),
        PROJECT_ROOT / raw.replace("/", "\\"),
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Arquivo não encontrado. Tentativas:\n"
        + "\n".join(str(p) for p in candidates)
    )


def read_channel_file(path: Path) -> np.ndarray:
    df = pd.read_csv(path)

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        raise ValueError(f"Sem coluna numérica: {path}")

    arr = numeric_df.to_numpy(dtype=np.float32).reshape(-1)

    return arr


def read_sample_5ch(row: pd.Series) -> np.ndarray:
    channels = []

    for i in range(1, 6):
        path = resolve_path(row[f"path_ch{i}"])
        ch = read_channel_file(path)
        channels.append(ch)

    min_len = min(len(ch) for ch in channels)

    if min_len < WINDOW_SIZE:
        raise ValueError("Amostra menor que WINDOW_SIZE.")

    channels = [ch[:min_len] for ch in channels]

    signal = np.stack(channels, axis=1).astype(np.float32)

    return signal


def iter_windows(signal: np.ndarray) -> Iterable[tuple[int, np.ndarray]]:
    n = signal.shape[0]
    start = 0

    while start < n:
        end = start + WINDOW_SIZE

        if end > n:
            if DROP_LAST:
                break

            window = signal[start:n]
            pad_len = WINDOW_SIZE - window.shape[0]
            window = np.pad(window, ((0, pad_len), (0, 0)), mode="constant")
        else:
            window = signal[start:end]

        yield start, window
        start += STEP


# =========================================================
# FEATURES
# =========================================================
def safe_div(a, b, eps=1e-12):
    return float(a / (b + eps))


def spectral_features(x: np.ndarray) -> dict[str, float]:
    eps = 1e-12

    x = x.astype(np.float64)
    x = x - np.mean(x)

    freqs = np.fft.rfftfreq(len(x), d=1.0 / FS)
    power = np.abs(np.fft.rfft(x)) ** 2

    total_power = float(np.sum(power) + eps)

    # ignora DC para pico dominante
    if len(power) > 1:
        peak_idx = int(np.argmax(power[1:]) + 1)
        dominant_freq = float(freqs[peak_idx])
        peak_power = float(power[peak_idx] / total_power)
    else:
        dominant_freq = 0.0
        peak_power = 0.0

    centroid = float(np.sum(freqs * power) / total_power)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))

    p = power / total_power
    entropy = float(-np.sum(p * np.log2(p + eps)) / np.log2(len(p) + eps))

    feats = {
        "dominant_freq": dominant_freq,
        "peak_power_ratio": peak_power,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_entropy": entropy,
    }

    band_values = {}

    for low, high in BANDS:
        mask = (freqs >= low) & (freqs < high)
        value = float(np.sum(power[mask]) / total_power)
        key = f"band_energy_{low}_{high}"
        feats[key] = value
        band_values[(low, high)] = value

    low_0_2k = (
        band_values.get((0, 500), 0.0)
        + band_values.get((500, 1000), 0.0)
        + band_values.get((1000, 2000), 0.0)
    )

    mid_2_7k = (
        band_values.get((2000, 4000), 0.0)
        + band_values.get((4000, 7000), 0.0)
    )

    high_9_11k = band_values.get((9000, 11000), 0.0)
    high_11_13k = band_values.get((11000, 13000), 0.0)

    feats["band_ratio_0_2000"] = float(low_0_2k)
    feats["band_ratio_2000_7000"] = float(mid_2_7k)
    feats["band_ratio_9000_11000"] = float(high_9_11k)
    feats["band_ratio_11000_13000"] = float(high_11_13k)
    feats["band_high_low_ratio"] = safe_div(high_9_11k, low_0_2k)
    feats["band_mid_low_ratio"] = safe_div(mid_2_7k, low_0_2k)

    return feats


def extract_features(window: np.ndarray) -> dict[str, float]:
    feats = {}

    for ch in range(window.shape[1]):
        x = window[:, ch].astype(np.float64)
        prefix = f"ch{ch + 1}"

        mean = float(np.mean(x))
        std = float(np.std(x))
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        rms = float(np.sqrt(np.mean(x ** 2)))
        p2p = float(xmax - xmin)
        abs_energy = float(np.sum(x ** 2))
        crest = safe_div(np.max(np.abs(x)), rms)

        feats[f"{prefix}_mean"] = mean
        feats[f"{prefix}_std"] = std
        feats[f"{prefix}_min"] = xmin
        feats[f"{prefix}_max"] = xmax
        feats[f"{prefix}_rms"] = rms
        feats[f"{prefix}_peak_to_peak"] = p2p
        feats[f"{prefix}_skewness"] = float(skew(x, bias=False, nan_policy="omit"))
        feats[f"{prefix}_kurtosis"] = float(kurtosis(x, fisher=True, bias=False, nan_policy="omit"))
        feats[f"{prefix}_crest_factor"] = crest
        feats[f"{prefix}_abs_energy"] = abs_energy

        spec = spectral_features(x)
        for name, value in spec.items():
            feats[f"{prefix}_{name}"] = value

    return feats


# =========================================================
# BUILD SPLIT
# =========================================================
def cap_healthy_train(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame):
    healthy_idx = np.where(y.values == "healthy")[0]
    other_idx = np.where(y.values != "healthy")[0]

    n_healthy = len(healthy_idx)

    if n_healthy <= MAX_HEALTHY_TRAIN_WINDOWS:
        print(f"[CAP] healthy treino mantido: {n_healthy}")
        return X, y, meta

    rng = np.random.default_rng(RANDOM_SEED)
    keep_healthy = rng.choice(
        healthy_idx,
        size=MAX_HEALTHY_TRAIN_WINDOWS,
        replace=False,
    )

    keep_idx = np.sort(np.concatenate([other_idx, keep_healthy]))

    print(f"\n[CAP] healthy treino: {n_healthy} -> {MAX_HEALTHY_TRAIN_WINDOWS}")

    X = X.iloc[keep_idx].reset_index(drop=True)
    y = y.iloc[keep_idx].reset_index(drop=True)
    meta = meta.iloc[keep_idx].reset_index(drop=True)

    return X, y, meta


def build_split_features(split_df: pd.DataFrame, split_name: str):
    rows = []
    labels = []
    meta = []

    print(f"\n[BUILD] {split_name} | amostras físicas: {len(split_df)}")

    for idx, row in split_df.reset_index(drop=True).iterrows():
        try:
            signal = read_sample_5ch(row)
        except Exception as e:
            print(f"[WARN] Falha lendo amostra {row.get('signal_id', '')}: {e}")
            continue

        rpm_value = int(row["rpm"])
        rpm_norm = rpm_value / 100.0

        for window_idx, (start_sample, window) in enumerate(iter_windows(signal)):
            feats = extract_features(window)
            feats["rpm_norm"] = rpm_norm

            rows.append(feats)
            labels.append(row["falha_base"])

            meta.append({
                "split": split_name,
                "signal_id": row["signal_id"],
                "grupo_condicao": row["grupo_condicao_mlp"],
                "falha_original": row["falha"],
                "falha_base": row["falha_base"],
                "rpm": row["rpm"],
                "window_idx": window_idx,
                "start_sample": start_sample,
            })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(split_df):
            print(f"[BUILD] {split_name}: {idx + 1}/{len(split_df)} amostras")

    if not rows:
        raise RuntimeError(f"Nenhuma feature gerada para {split_name}")

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="label")
    meta = pd.DataFrame(meta)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if split_name == "train":
        X, y, meta = cap_healthy_train(X, y, meta)

    return X, y, meta


# =========================================================
# SAVE
# =========================================================
def save_outputs(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    meta_train,
    meta_val,
    meta_test,
    elapsed,
):
    feature_names = sorted(X_train.columns)

    X_train = X_train[feature_names]
    X_val = X_val.reindex(columns=feature_names, fill_value=0.0)
    X_test = X_test.reindex(columns=feature_names, fill_value=0.0)

    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_val.to_csv(OUTPUT_DIR / "X_val.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_val.to_csv(OUTPUT_DIR / "y_val.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    meta_train.to_csv(OUTPUT_DIR / "meta_train.csv", index=False)
    meta_val.to_csv(OUTPUT_DIR / "meta_val.csv", index=False)
    meta_test.to_csv(OUTPUT_DIR / "meta_test.csv", index=False)

    np.savez_compressed(OUTPUT_DIR / "X_train.npz", X=X_train.to_numpy(np.float32))
    np.savez_compressed(OUTPUT_DIR / "X_val.npz", X=X_val.to_numpy(np.float32))
    np.savez_compressed(OUTPUT_DIR / "X_test.npz", X=X_test.to_numpy(np.float32))

    pd.Series(feature_names, name="feature").to_csv(
        OUTPUT_DIR / "feature_names.csv",
        index=False,
    )

    class_dist = pd.DataFrame({
        "train": y_train.value_counts(),
        "val": y_val.value_counts(),
        "test": y_test.value_counts(),
    }).fillna(0).astype(int)

    class_dist["total"] = class_dist.sum(axis=1)
    class_dist.index.name = "classe"
    class_dist.reset_index().to_csv(OUTPUT_DIR / "class_distribution.csv", index=False)

    feature_summary = pd.DataFrame({
        "feature": feature_names,
        "mean": X_train.mean(axis=0).values,
        "std": X_train.std(axis=0).values,
        "min": X_train.min(axis=0).values,
        "max": X_train.max(axis=0).values,
        "n_nan": X_train.isna().sum(axis=0).values,
    })
    feature_summary.to_csv(OUTPUT_DIR / "feature_summary.csv", index=False)

    full_df = pd.concat([
        pd.concat([X_train, y_train, meta_train], axis=1),
        pd.concat([X_val, y_val, meta_val], axis=1),
        pd.concat([X_test, y_test, meta_test], axis=1),
    ], axis=0).reset_index(drop=True)

    try:
        full_df.to_parquet(OUTPUT_DIR / "dataset_full.parquet", index=False)
    except Exception:
        full_df.to_csv(OUTPUT_DIR / "dataset_full.csv", index=False)

    summary = {
        "catalog_path": str(CATALOG_PATH),
        "output_dir": str(OUTPUT_DIR),
        "sensor": "Vibration",
        "channels": 5,
        "window_size": WINDOW_SIZE,
        "overlap": OVERLAP,
        "step": STEP,
        "fs": FS,
        "n_features": len(feature_names),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_classes": y_train.nunique(),
        "classes": "|".join(sorted(y_train.unique())),
        "target_labels": "|".join(TARGET_LABELS),
        "excluded_labels": "|".join(EXCLUDED_LABELS),
        "healthy_train_cap": MAX_HEALTHY_TRAIN_WINDOWS,
        "elapsed_sec": elapsed,
    }

    pd.Series(summary).to_csv(OUTPUT_DIR / "dataset_summary.csv")


# =========================================================
# MAIN
# =========================================================
def main():
    start = time.time()
    np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print("BUILD MLP CACHE — VIBRATION ONLY / NO BPFI-BPFO")
    print("=" * 70)
    print("CATALOG:", CATALOG_PATH)
    print("OUTPUT:", OUTPUT_DIR)
    print("=" * 70)

    df = load_catalog()

    df.to_csv(OUTPUT_DIR / "catalog_mlp_filtered.csv", index=False)

    print("\n[CATALOG]")
    print("amostras:", len(df))
    print("classes:", sorted(df["falha_base"].unique()))

    print("\nDistribuição:")
    print(df["falha_base"].value_counts().sort_index())

    print("\nGrupos:")
    print(df.groupby("falha_base")["grupo_condicao_mlp"].nunique().sort_index())

    outer_splits = stratified_group_split(
        df=df,
        group_col="grupo_condicao_mlp",
        label_col="falha_base",
        n_splits=OUTER_N_SPLITS,
        random_state=OUTER_RANDOM_STATE,
    )

    train_df, test_df = outer_splits[OUTER_FOLD_INDEX]

    train_df, val_df = split_train_val_by_group(
        train_df=train_df,
        group_col="grupo_condicao_mlp",
        label_col="falha_base",
        val_size=VAL_SIZE,
        random_state=VAL_RANDOM_STATE,
    )

    print("\n[SPLIT]")
    print("train:", len(train_df), train_df["falha_base"].value_counts().to_dict())
    print("val:  ", len(val_df), val_df["falha_base"].value_counts().to_dict())
    print("test: ", len(test_df), test_df["falha_base"].value_counts().to_dict())

    print("\n[SPLIT RPM x CLASSE]")
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n{name}:")
        print(sdf.groupby(["falha_base", "rpm"]).size().to_string())

    save_split_audit(train_df, val_df, test_df)

    X_train, y_train, meta_train = build_split_features(train_df, "train")
    X_val, y_val, meta_val = build_split_features(val_df, "val")
    X_test, y_test, meta_test = build_split_features(test_df, "test")

    elapsed = time.time() - start

    save_outputs(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        meta_train,
        meta_val,
        meta_test,
        elapsed,
    )

    print("\n" + "=" * 70)
    print("CACHE MLP GERADO COM SUCESSO")
    print("=" * 70)
    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("X_test: ", X_test.shape)
    print(f"tempo: {elapsed:.2f}s ({elapsed / 60:.2f} min)")
    print("saída:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()