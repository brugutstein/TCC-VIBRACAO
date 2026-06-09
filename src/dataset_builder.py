from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

from src.ingest import window_csv_file, window_vibration_sample
from src.preprocess import fit_transform_train_test


DEFAULT_STAT_FEATURES = [
    "mean",
    "std",
    "max",
    "min",
    "rms",
    "peak_to_peak",
    "crest_factor",
    "skew",
    "kurtosis",
    "abs_energy",
]

OPTIONAL_STAT_FEATURES = []

# =========================
# STAT FEATURES (V0-LIKE POR JANELA)
# =========================
def _safe_skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    std = x.std()
    if std < 1e-12:
        return 0.0
    centered = x - x.mean()
    return float(np.mean(centered ** 3) / (std ** 3 + 1e-12))


def _safe_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    std = x.std()
    if std < 1e-12:
        return 0.0
    centered = x - x.mean()
    return float(np.mean(centered ** 4) / (std ** 4 + 1e-12))


def get_stat_feature_names(
    include_optional: bool = False,
    stats_mode: str = "v0_window",
) -> list[str]:
    if stats_mode == "v0_window":
        return list(DEFAULT_STAT_FEATURES)

    names = list(DEFAULT_STAT_FEATURES)
    if include_optional:
        names.extend(OPTIONAL_STAT_FEATURES)
    return names


def extract_spectral_features_from_window(
    window: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    if not isinstance(window, np.ndarray):
        window = np.asarray(window)

    if window.ndim == 1:
        window = window[:, np.newaxis]

    if window.ndim != 2:
        raise ValueError(f"Janela esperada como 1D/2D, recebido shape {window.shape}")

    features: list[float] = []
    feature_names: list[str] = []

    for ch_idx in range(window.shape[1]):
        x = window[:, ch_idx].astype(np.float64)

        spectrum = np.abs(np.fft.rfft(x))
        spectrum = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)

        power = spectrum ** 2
        power_sum = float(np.sum(power)) + 1e-12

        freqs = np.fft.rfftfreq(len(x), d=1.0)

        spectrum_no_dc = spectrum.copy()
        power_no_dc = power.copy()
        if len(spectrum_no_dc) > 0:
            spectrum_no_dc[0] = 0.0
            power_no_dc[0] = 0.0

        power_no_dc_sum = float(np.sum(power_no_dc)) + 1e-12

        dominant_idx = int(np.argmax(spectrum_no_dc)) if len(spectrum_no_dc) > 0 else 0
        dominant_freq = float(freqs[dominant_idx]) if len(freqs) > dominant_idx else 0.0

        spectral_centroid = float(np.sum(freqs * power) / power_sum)

        spectral_bandwidth = float(
            np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power) / power_sum)
        )

        cumulative_power = np.cumsum(power)
        rolloff_threshold = 0.85 * cumulative_power[-1] if len(cumulative_power) > 0 else 0.0
        rolloff_idx = int(np.searchsorted(cumulative_power, rolloff_threshold)) if len(cumulative_power) > 0 else 0
        rolloff_idx = min(rolloff_idx, len(freqs) - 1) if len(freqs) > 0 else 0
        spectral_rolloff = float(freqs[rolloff_idx]) if len(freqs) > 0 else 0.0

        p = power_no_dc / power_no_dc_sum
        spectral_entropy = float(-np.sum(p * np.log2(p + 1e-12)))

        positive_spec = spectrum_no_dc[spectrum_no_dc > 0]
        if len(positive_spec) == 0:
            spectral_flatness = 0.0
        else:
            gm = float(np.exp(np.mean(np.log(positive_spec + 1e-12))))
            am = float(np.mean(spectrum_no_dc + 1e-12))
            spectral_flatness = gm / (am + 1e-12)

        max_freq = freqs.max() if len(freqs) > 0 else 0.0
        band_defs = [
            ("spec_band_0", 0.00, 0.10),
            ("spec_band_1", 0.10, 0.20),
            ("spec_band_2", 0.20, 0.40),
            ("spec_band_3", 0.40, 0.50),
        ]

        band_values = {}
        for band_name, low, high in band_defs:
            if max_freq == 0.0:
                band_energy = 0.0
            else:
                if high == 0.50:
                    mask = (freqs >= low * max_freq) & (freqs <= high * max_freq)
                else:
                    mask = (freqs >= low * max_freq) & (freqs < high * max_freq)
                band_energy = float(np.sum(power[mask]) / power_sum)
            band_values[band_name] = band_energy

        values_map = {
            "dominant_freq": dominant_freq,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_rolloff": spectral_rolloff,
            "spectral_entropy": spectral_entropy,
            "spectral_flatness": spectral_flatness,
            **band_values,
        }

        for name, value in values_map.items():
            features.append(value)
            feature_names.append(f"ch{ch_idx}_{name}")

    features_array = np.nan_to_num(
        np.asarray(features, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return features_array, feature_names


def extract_v0_features_from_window(
    window: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    if not isinstance(window, np.ndarray):
        window = np.asarray(window)

    if window.ndim == 1:
        window = window[:, np.newaxis]

    if window.ndim != 2:
        raise ValueError(f"Janela esperada como 1D/2D, recebido shape {window.shape}")

    stat_names = get_stat_feature_names(stats_mode="v0_window")
    features: list[float] = []
    feature_names: list[str] = []

    for ch_idx in range(window.shape[1]):
        x = window[:, ch_idx].astype(np.float64)

        mean_ = float(np.mean(x))
        std_ = float(np.std(x))
        max_ = float(np.max(x))
        min_ = float(np.min(x))
        rms_ = float(np.sqrt(np.mean(np.square(x))))
        peak_to_peak_ = float(np.ptp(x))
        crest_factor_ = float(np.max(np.abs(x)) / (rms_ + 1e-12))
        skew_ = _safe_skew(x)
        kurtosis_ = _safe_kurtosis(x)
        abs_energy_ = float(np.sum(np.square(x)))

        values_map = {
            "mean": mean_,
            "std": std_,
            "max": max_,
            "min": min_,
            "rms": rms_,
            "peak_to_peak": peak_to_peak_,
            "crest_factor": crest_factor_,
            "skew": skew_,
            "kurtosis": kurtosis_,
            "abs_energy": abs_energy_,
        }

        for stat_name in stat_names:
            features.append(values_map[stat_name])
            feature_names.append(f"ch{ch_idx}_{stat_name}")

    features_array = np.nan_to_num(
        np.asarray(features, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return features_array, feature_names


def extract_stat_features_from_window(
    window: np.ndarray,
    include_optional: bool = False,
    stats_mode: str = "v0_window",
) -> tuple[np.ndarray, list[str]]:
    if stats_mode == "v0_window":
        return extract_v0_features_from_window(window)

    if stats_mode == "v0_plus_spectral":
        v0_features, v0_names = extract_v0_features_from_window(window)
        spec_features, spec_names = extract_spectral_features_from_window(window)

        features = np.concatenate([v0_features, spec_features], axis=0).astype(np.float32)
        feature_names = v0_names + spec_names
        return features, feature_names

    raise ValueError(f"stats_mode inválido: {stats_mode}")


def extract_stat_features_from_windows(
    windows: np.ndarray,
    include_optional: bool = False,
    stats_mode: str = "v0_window",
) -> tuple[np.ndarray, list[str]]:
    if not isinstance(windows, np.ndarray):
        windows = np.asarray(windows)

    if windows.ndim == 2:
        windows = windows[..., np.newaxis]

    if windows.ndim != 3:
        raise ValueError(f"Esperado windows 2D/3D, recebido shape {windows.shape}")

    stats_rows: list[np.ndarray] = []
    stats_feature_names: list[str] | None = None

    for window in windows:
        row, feature_names = extract_stat_features_from_window(
            window=window,
            include_optional=include_optional,
            stats_mode=stats_mode,
        )
        stats_rows.append(row)
        if stats_feature_names is None:
            stats_feature_names = feature_names

    if not stats_rows:
        n_channels = windows.shape[-1]
        stat_names = get_stat_feature_names(
            include_optional=include_optional,
            stats_mode=stats_mode,
        )
        n_features = n_channels * len(stat_names)
        stats_feature_names = [
            f"ch{ch}_{name}" for ch in range(n_channels) for name in stat_names
        ]
        return np.empty((0, n_features), dtype=np.float32), stats_feature_names

    return np.stack(stats_rows, axis=0).astype(np.float32), stats_feature_names or []


def compute_band_energy_from_window(window: np.ndarray) -> np.ndarray:
    if not isinstance(window, np.ndarray):
        window = np.asarray(window)

    if window.ndim == 1:
        window = window[:, np.newaxis]

    if window.ndim != 2:
        raise ValueError(f"Janela esperada como 1D/2D para band energy, recebido shape {window.shape}")

    energies: list[float] = []

    for ch_idx in range(window.shape[1]):
        x = window[:, ch_idx].astype(np.float64)

        fft_vals = np.fft.rfft(x)
        fft_mag = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(len(x), d=1.0)
        max_freq = freqs.max() if len(freqs) > 0 else 0.0

        bands = [
            (0.0, 0.1),
            (0.1, 0.2),
            (0.2, 0.4),
            (0.4, 0.5),
        ]

        for band_idx, (low, high) in enumerate(bands):
            if max_freq == 0.0:
                energy = 0.0
            else:
                if band_idx == len(bands) - 1:
                    mask = (freqs >= low * max_freq) & (freqs <= high * max_freq)
                else:
                    mask = (freqs >= low * max_freq) & (freqs < high * max_freq)
                energy = float(np.sum(fft_mag[mask] ** 2))
            energies.append(energy)

    return np.nan_to_num(np.asarray(energies, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# =========================
# DATASET BUILDERS
# =========================
def _looks_like_vibration_sample_catalog(df_subset: pd.DataFrame) -> bool:
    required = {"sensor", "path_ch1", "path_ch2", "path_ch3", "path_ch4", "path_ch5"}
    return required.issubset(df_subset.columns)


def build_dataset_from_catalog(
    df_subset: pd.DataFrame,
    label_col: str = "falha",
    selected_columns: Optional[list[str]] = None,
    window_size: int = 1024,
    overlap: float = 0.5,
    drop_last: bool = True,
    nrows: Optional[int] = None,
    return_stats: bool = False,
    stats_include_optional: bool = False,
    stats_mode: str = "v0_plus_spectral",
    include_band_energy: bool = True,
):
    """
    Aceita dois formatos:

    1) catálogo antigo por arquivo:
       precisa ter coluna 'caminho'

    2) catálogo novo consolidado de vibração:
       precisa ter path_ch1..path_ch5
    """
    X_list: list[np.ndarray] = []
    X_stats_list: list[np.ndarray] = []
    y_list: list[str] = []
    meta_rows: list[dict] = []
    stats_feature_names: list[str] | None = None

    if _looks_like_vibration_sample_catalog(df_subset):
        required_cols = {label_col, "sensor", "grupo_condicao", "path_ch1", "path_ch2", "path_ch3", "path_ch4", "path_ch5"}
        missing_cols = required_cols - set(df_subset.columns)
        if missing_cols:
            raise ValueError(f"Colunas ausentes em df_subset: {missing_cols}")

        iterator_desc = "Processando amostras consolidadas"

        for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=iterator_desc):
            label_value = row[label_col]

            windows, meta_df_sample, metadata = window_vibration_sample(
                sample_row=row,
                window_size=window_size,
                overlap=overlap,
                drop_last=drop_last,
                selected_columns=selected_columns,
                nrows=nrows,
            )

            if len(windows) == 0:
                continue

            if windows.ndim == 2:
                windows = windows[..., np.newaxis]

            X_list.append(windows)
            y_list.extend([label_value] * len(windows))

            if return_stats:
                windows_stats, current_feature_names = extract_stat_features_from_windows(
                    windows=windows,
                    include_optional=stats_include_optional,
                    stats_mode=stats_mode,
                )

                if include_band_energy:
                    band_energy_rows: list[np.ndarray] = []
                    for window in windows:
                        band_energy_rows.append(compute_band_energy_from_window(window))

                    band_energy_array = np.stack(band_energy_rows, axis=0).astype(np.float32)
                    windows_stats = np.concatenate([windows_stats, band_energy_array], axis=1)

                    band_names: list[str] = []
                    n_channels = windows.shape[-1]
                    for ch in range(n_channels):
                        for i in range(4):
                            band_names.append(f"ch{ch}_band_energy_{i}")

                    current_feature_names = current_feature_names + band_names

                X_stats_list.append(windows_stats)

                if stats_feature_names is None:
                    stats_feature_names = current_feature_names
                elif stats_feature_names != current_feature_names:
                    raise RuntimeError("Inconsistência interna: nomes de stats variaram entre amostras.")

            n_stats_features = len(stats_feature_names) if stats_feature_names is not None else 0

            for _, meta_row in meta_df_sample.iterrows():
                meta_rows.append({
                    "arquivo": "",
                    "caminho": "",
                    "sensor": row.get("sensor", ""),
                    "equipamento": row.get("equipamento", ""),
                    "rpm": row.get("rpm", ""),
                    "falha": row.get("falha", ""),
                    "canal_arquivo": "",
                    "grupo_condicao": row.get("grupo_condicao", ""),
                    "signal_id": row.get("signal_id", ""),
                    "segment_index": int(meta_row.get("segment_index", -1)),
                    "segment_name": str(meta_row.get("segment_name", "")),
                    "window_index": int(meta_row.get("window_index", 0)),
                    "n_signal_columns": int(metadata.get("n_channels", windows.shape[-1])),
                    "n_stats_features": int(n_stats_features),
                })
    else:
        required_cols = {"caminho", label_col}
        missing_cols = required_cols - set(df_subset.columns)
        if missing_cols:
            raise ValueError(f"Colunas ausentes em df_subset: {missing_cols}")

        iterator_desc = "Processando arquivos"

        for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=iterator_desc):
            file_path = row["caminho"]
            label_value = row[label_col]

            windows, metadata = window_csv_file(
                file_path=file_path,
                selected_columns=selected_columns,
                window_size=window_size,
                overlap=overlap,
                drop_last=drop_last,
                nrows=nrows,
            )

            if len(windows) == 0:
                continue

            if windows.ndim == 2:
                windows = windows[..., np.newaxis]

            X_list.append(windows)
            y_list.extend([label_value] * len(windows))

            if return_stats:
                windows_stats, current_feature_names = extract_stat_features_from_windows(
                    windows=windows,
                    include_optional=stats_include_optional,
                    stats_mode=stats_mode,
                )

                if include_band_energy:
                    band_energy_rows: list[np.ndarray] = []
                    for window in windows:
                        band_energy_rows.append(compute_band_energy_from_window(window))

                    band_energy_array = np.stack(band_energy_rows, axis=0).astype(np.float32)
                    windows_stats = np.concatenate([windows_stats, band_energy_array], axis=1)

                    band_names: list[str] = []
                    n_channels = windows.shape[-1]
                    for ch in range(n_channels):
                        for i in range(4):
                            band_names.append(f"ch{ch}_band_energy_{i}")

                    current_feature_names = current_feature_names + band_names

                X_stats_list.append(windows_stats)

                if stats_feature_names is None:
                    stats_feature_names = current_feature_names
                elif stats_feature_names != current_feature_names:
                    raise RuntimeError("Inconsistência interna: nomes de stats variaram entre arquivos.")

            n_stats_features = len(stats_feature_names) if stats_feature_names is not None else 0

            for window_idx in range(len(windows)):
                meta_rows.append({
                    "arquivo": row.get("arquivo", ""),
                    "caminho": file_path,
                    "sensor": row.get("sensor", ""),
                    "equipamento": row.get("equipamento", ""),
                    "rpm": row.get("rpm", ""),
                    "falha": row.get("falha", ""),
                    "canal_arquivo": row.get("canal_arquivo", ""),
                    "grupo_condicao": row.get("grupo_condicao", ""),
                    "signal_id": row.get("signal_id", ""),
                    "segment_index": -1,
                    "segment_name": "",
                    "window_index": int(window_idx),
                    "n_signal_columns": int(metadata.get("n_signal_columns", windows.shape[-1])),
                    "n_stats_features": int(n_stats_features),
                })

    if not X_list:
        raise ValueError("Nenhuma janela foi gerada. Verifique os arquivos e parâmetros.")

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list)
    meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)

    if len(X) != len(y) or len(X) != len(meta_df):
        raise RuntimeError("Inconsistência interna: X, y e meta_df possuem tamanhos diferentes.")

    if not return_stats:
        return X, y, meta_df

    if not X_stats_list:
        raise RuntimeError("return_stats=True, mas nenhuma feature estatística foi gerada.")

    X_stats = np.concatenate(X_stats_list, axis=0).astype(np.float32)

    if len(X_stats) != len(X):
        raise RuntimeError("Inconsistência interna: X_stats e X possuem tamanhos diferentes.")

    return X, X_stats, y, meta_df, (stats_feature_names or [])


# =========================
# LABEL ENCODING
# =========================
def encode_labels(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)
    return y_train_enc, y_test_enc, encoder


# =========================
# SEQUENCE PACKING
# =========================
def _validate_pack_inputs(
    X: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    seq_len: int,
    sequence_stride: Optional[int],
    grouping_col: str,
) -> int:
    if X.ndim != 3:
        raise ValueError(f"Esperado X 3D antes do empacotamento, recebido shape {X.shape}")

    if len(X) != len(y) or len(X) != len(meta_df):
        raise ValueError("X, y e meta_df devem ter o mesmo número de amostras.")

    if seq_len <= 0:
        raise ValueError("seq_len deve ser > 0")

    if sequence_stride is None:
        sequence_stride = seq_len

    if sequence_stride <= 0:
        raise ValueError("sequence_stride deve ser > 0")

    required_meta_cols = {grouping_col, "window_index"}
    missing_meta_cols = required_meta_cols - set(meta_df.columns)
    if missing_meta_cols:
        raise ValueError(f"Colunas ausentes em meta_df para empacotamento: {missing_meta_cols}")

    return sequence_stride


def _pack_common_indices(
    y: np.ndarray,
    meta_df: pd.DataFrame,
    seq_len: int,
    sequence_stride: Optional[int],
    drop_incomplete: bool,
    grouping_col: str,
):
    fake_X = np.zeros((len(y), 1, 1), dtype=np.float32)

    sequence_stride = _validate_pack_inputs(
        X=fake_X,
        y=y,
        meta_df=meta_df,
        seq_len=seq_len,
        sequence_stride=sequence_stride,
        grouping_col=grouping_col,
    )

    work_df = meta_df.copy().reset_index(drop=True)
    work_df["_sample_idx"] = np.arange(len(work_df))
    work_df["_label"] = y

    chunks: list[dict] = []

    group_iter = work_df.groupby(grouping_col, sort=False)

    for group_value, group_df in tqdm(
        group_iter,
        total=work_df[grouping_col].nunique(),
        desc=f"Empacotando sequências ({grouping_col})",
    ):
        sort_cols = ["window_index"]
        if "segment_index" in group_df.columns:
            sort_cols = ["segment_index", "window_index"]

        group_df = group_df.sort_values(sort_cols).reset_index(drop=True)

        indices = group_df["_sample_idx"].to_numpy()
        labels = group_df["_label"].to_numpy()
        window_indices = group_df["window_index"].to_numpy()

        n_windows = len(group_df)
        if n_windows == 0:
            continue

        for start in range(0, n_windows, sequence_stride):
            end = start + seq_len

            if end > n_windows:
                if drop_incomplete:
                    break
                continue

            chunk_indices = indices[start:end]
            chunk_labels = labels[start:end]

            unique_labels = np.unique(chunk_labels)
            if len(unique_labels) != 1:
                raise ValueError(
                    f"Sequência com múltiplos labels encontrada em {grouping_col}={group_value}: "
                    f"{unique_labels.tolist()}"
                )

            first_row = group_df.iloc[start]
            last_row = group_df.iloc[end - 1]

            chunks.append({
                "indices": chunk_indices,
                "label": unique_labels[0],
                "meta": {
                    "sequence_id": len(chunks),
                    grouping_col: group_value,
                    "arquivo": first_row.get("arquivo", ""),
                    "caminho": first_row.get("caminho", ""),
                    "sensor": first_row.get("sensor", ""),
                    "equipamento": first_row.get("equipamento", ""),
                    "rpm": first_row.get("rpm", ""),
                    "falha": first_row.get("falha", ""),
                    "canal_arquivo": first_row.get("canal_arquivo", ""),
                    "grupo_condicao": first_row.get("grupo_condicao", ""),
                    "signal_id": first_row.get("signal_id", ""),
                    "seq_len": int(seq_len),
                    "sequence_stride": int(sequence_stride),
                    "start_window_index": int(first_row["window_index"]),
                    "end_window_index": int(last_row["window_index"]),
                    "n_windows_in_sequence": int(end - start),
                },
            })

    if not chunks:
        raise ValueError(
            "Nenhuma sequência foi gerada. "
            "Verifique seq_len, sequence_stride, drop_incomplete e a quantidade de janelas por grupo."
        )

    return chunks, sequence_stride


def pack_windows_into_sequences(
    X: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    seq_len: int,
    sequence_stride: Optional[int] = None,
    drop_incomplete: bool = True,
    grouping_col: str = "grupo_condicao",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    chunks, _ = _pack_common_indices(
        y=y,
        meta_df=meta_df,
        seq_len=seq_len,
        sequence_stride=sequence_stride,
        drop_incomplete=drop_incomplete,
        grouping_col=grouping_col,
    )

    X_seq = np.stack([X[item["indices"]] for item in chunks], axis=0).astype(np.float32)
    y_seq = np.asarray([item["label"] for item in chunks])
    meta_seq_df = pd.DataFrame([item["meta"] for item in chunks]).reset_index(drop=True)

    if len(X_seq) != len(y_seq) or len(X_seq) != len(meta_seq_df):
        raise RuntimeError("Inconsistência interna: X_seq, y_seq e meta_seq_df possuem tamanhos diferentes.")

    return X_seq, y_seq, meta_seq_df


def pack_windows_and_stats_into_sequences(
    X: np.ndarray,
    X_stats: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    seq_len: int,
    sequence_stride: Optional[int] = None,
    drop_incomplete: bool = True,
    grouping_col: str = "grupo_condicao",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    if X.ndim != 3:
        raise ValueError(f"Esperado X 3D antes do empacotamento, recebido shape {X.shape}")

    if X_stats.ndim != 2:
        raise ValueError(f"Esperado X_stats 2D antes do empacotamento, recebido shape {X_stats.shape}")

    if len(X_stats) != len(X):
        raise ValueError("X_stats deve ter o mesmo número de janelas que X.")

    if len(X) != len(y) or len(X) != len(meta_df):
        raise ValueError("X, y e meta_df devem ter o mesmo número de amostras.")

    chunks, _ = _pack_common_indices(
        y=y,
        meta_df=meta_df,
        seq_len=seq_len,
        sequence_stride=sequence_stride,
        drop_incomplete=drop_incomplete,
        grouping_col=grouping_col,
    )

    X_seq = np.stack([X[item["indices"]] for item in chunks], axis=0).astype(np.float32)
    X_stats_seq = np.stack([X_stats[item["indices"]] for item in chunks], axis=0).astype(np.float32)
    y_seq = np.asarray([item["label"] for item in chunks])
    meta_seq_df = pd.DataFrame([item["meta"] for item in chunks]).reset_index(drop=True)

    if len(X_seq) != len(X_stats_seq) or len(X_seq) != len(y_seq) or len(X_seq) != len(meta_seq_df):
        raise RuntimeError(
            "Inconsistência interna: X_seq, X_stats_seq, y_seq e meta_seq_df possuem tamanhos diferentes."
        )

    return X_seq, X_stats_seq, y_seq, meta_seq_df


# =========================
# SCALING
# =========================
def _scale_sequence_data(
    X_train_seq: np.ndarray,
    X_test_seq: np.ndarray,
):
    if X_train_seq.ndim != 4 or X_test_seq.ndim != 4:
        raise ValueError(
            "Esperado arrays 4D para scaling de sequências: "
            f"train={X_train_seq.shape}, test={X_test_seq.shape}"
        )

    train_shape = X_train_seq.shape
    test_shape = X_test_seq.shape

    X_train_flat = X_train_seq.reshape(-1, train_shape[2], train_shape[3])
    X_test_flat = X_test_seq.reshape(-1, test_shape[2], test_shape[3])

    X_train_scaled, X_test_scaled, scaler = fit_transform_train_test(X_train_flat, X_test_flat)

    X_train_scaled = X_train_scaled.reshape(train_shape).astype(np.float32)
    X_test_scaled = X_test_scaled.reshape(test_shape).astype(np.float32)

    return X_train_scaled, X_test_scaled, scaler


def _scale_stat_data(
    X_train_stats: np.ndarray,
    X_test_stats: np.ndarray,
):
    if X_train_stats.ndim not in (2, 3) or X_test_stats.ndim not in (2, 3):
        raise ValueError(
            "Esperado arrays 2D/3D para scaling de stats: "
            f"train={X_train_stats.shape}, test={X_test_stats.shape}"
        )

    train_shape = X_train_stats.shape
    test_shape = X_test_stats.shape

    X_train_2d = X_train_stats.reshape(-1, train_shape[-1])
    X_test_2d = X_test_stats.reshape(-1, test_shape[-1])

    scaler = StandardScaler()
    scaler.fit(X_train_2d)

    X_train_scaled = scaler.transform(X_train_2d).reshape(train_shape).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_2d).reshape(test_shape).astype(np.float32)

    return X_train_scaled, X_test_scaled, scaler


# =========================
# FOLD BUILDERS
# =========================
def build_fold_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "falha",
    selected_columns: Optional[list[str]] = None,
    window_size: int = 1024,
    overlap: float = 0.5,
    drop_last: bool = True,
    nrows: Optional[int] = None,
    apply_scaling: bool = True,
    return_stats: bool = False,
    stats_include_optional: bool = False,
):
    if not return_stats:
        X_train, y_train, meta_train = build_dataset_from_catalog(
            df_subset=train_df,
            label_col=label_col,
            selected_columns=selected_columns,
            window_size=window_size,
            overlap=overlap,
            drop_last=drop_last,
            nrows=nrows,
            return_stats=False,
        )

        X_test, y_test, meta_test = build_dataset_from_catalog(
            df_subset=test_df,
            label_col=label_col,
            selected_columns=selected_columns,
            window_size=window_size,
            overlap=overlap,
            drop_last=drop_last,
            nrows=nrows,
            return_stats=False,
        )

        y_train_enc, y_test_enc, label_encoder = encode_labels(y_train, y_test)

        scaler = None
        if apply_scaling:
            X_train, X_test, scaler = fit_transform_train_test(X_train, X_test)

        return {
            "mode": "window",
            "X_train": X_train,
            "y_train": y_train_enc,
            "X_test": X_test,
            "y_test": y_test_enc,
            "y_train_raw": y_train,
            "y_test_raw": y_test,
            "meta_train": meta_train,
            "meta_test": meta_test,
            "label_encoder": label_encoder,
            "scaler": scaler,
        }

    X_train, X_train_stats, y_train, meta_train, stats_feature_names = build_dataset_from_catalog(
        df_subset=train_df,
        label_col=label_col,
        selected_columns=selected_columns,
        window_size=window_size,
        overlap=overlap,
        drop_last=drop_last,
        nrows=nrows,
        return_stats=True,
        stats_include_optional=stats_include_optional,
    )

    X_test, X_test_stats, y_test, meta_test, _ = build_dataset_from_catalog(
        df_subset=test_df,
        label_col=label_col,
        selected_columns=selected_columns,
        window_size=window_size,
        overlap=overlap,
        drop_last=drop_last,
        nrows=nrows,
        return_stats=True,
        stats_include_optional=stats_include_optional,
    )

    y_train_enc, y_test_enc, label_encoder = encode_labels(y_train, y_test)

    signal_scaler = None
    stats_scaler = None
    if apply_scaling:
        X_train, X_test, signal_scaler = fit_transform_train_test(X_train, X_test)
        X_train_stats, X_test_stats, stats_scaler = _scale_stat_data(X_train_stats, X_test_stats)

    return {
        "mode": "window_with_stats",
        "X_train": X_train,
        "X_train_stats": X_train_stats,
        "y_train": y_train_enc,
        "X_test": X_test,
        "X_test_stats": X_test_stats,
        "y_test": y_test_enc,
        "y_train_raw": y_train,
        "y_test_raw": y_test,
        "meta_train": meta_train,
        "meta_test": meta_test,
        "label_encoder": label_encoder,
        "scaler": signal_scaler,
        "stats_scaler": stats_scaler,
        "stats_feature_names": stats_feature_names,
    }


def build_fold_sequence_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "falha",
    selected_columns: Optional[list[str]] = None,
    window_size: int = 1024,
    overlap: float = 0.5,
    drop_last: bool = True,
    nrows: Optional[int] = None,
    apply_scaling: bool = True,
    seq_len: int = 10,
    sequence_stride: Optional[int] = None,
    drop_incomplete_sequences: bool = True,
    grouping_col: str = "grupo_condicao",
    return_stats: bool = False,
    stats_include_optional: bool = False,
):
    window_data = build_fold_data(
        train_df=train_df,
        test_df=test_df,
        label_col=label_col,
        selected_columns=selected_columns,
        window_size=window_size,
        overlap=overlap,
        drop_last=drop_last,
        nrows=nrows,
        apply_scaling=False,
        return_stats=return_stats,
        stats_include_optional=stats_include_optional,
    )

    if not return_stats:
        X_train_seq, y_train_seq, meta_train_seq = pack_windows_into_sequences(
            X=window_data["X_train"],
            y=window_data["y_train_raw"],
            meta_df=window_data["meta_train"],
            seq_len=seq_len,
            sequence_stride=sequence_stride,
            drop_incomplete=drop_incomplete_sequences,
            grouping_col=grouping_col,
        )

        X_test_seq, y_test_seq, meta_test_seq = pack_windows_into_sequences(
            X=window_data["X_test"],
            y=window_data["y_test_raw"],
            meta_df=window_data["meta_test"],
            seq_len=seq_len,
            sequence_stride=sequence_stride,
            drop_incomplete=drop_incomplete_sequences,
            grouping_col=grouping_col,
        )

        y_train_enc, y_test_enc, label_encoder = encode_labels(y_train_seq, y_test_seq)

        scaler = None
        if apply_scaling:
            X_train_seq, X_test_seq, scaler = _scale_sequence_data(X_train_seq, X_test_seq)

        return {
            "mode": "sequence",
            "X_train": X_train_seq,
            "y_train": y_train_enc,
            "X_test": X_test_seq,
            "y_test": y_test_enc,
            "y_train_raw": y_train_seq,
            "y_test_raw": y_test_seq,
            "meta_train": meta_train_seq,
            "meta_test": meta_test_seq,
            "label_encoder": label_encoder,
            "scaler": scaler,
            "seq_len": seq_len,
            "sequence_stride": sequence_stride if sequence_stride is not None else seq_len,
            "grouping_col": grouping_col,
        }

    X_train_seq, X_train_stats_seq, y_train_seq, meta_train_seq = pack_windows_and_stats_into_sequences(
        X=window_data["X_train"],
        X_stats=window_data["X_train_stats"],
        y=window_data["y_train_raw"],
        meta_df=window_data["meta_train"],
        seq_len=seq_len,
        sequence_stride=sequence_stride,
        drop_incomplete=drop_incomplete_sequences,
        grouping_col=grouping_col,
    )

    X_test_seq, X_test_stats_seq, y_test_seq, meta_test_seq = pack_windows_and_stats_into_sequences(
        X=window_data["X_test"],
        X_stats=window_data["X_test_stats"],
        y=window_data["y_test_raw"],
        meta_df=window_data["meta_test"],
        seq_len=seq_len,
        sequence_stride=sequence_stride,
        drop_incomplete=drop_incomplete_sequences,
        grouping_col=grouping_col,
    )

    y_train_enc, y_test_enc, label_encoder = encode_labels(y_train_seq, y_test_seq)

    signal_scaler = None
    stats_scaler = None
    if apply_scaling:
        X_train_seq, X_test_seq, signal_scaler = _scale_sequence_data(X_train_seq, X_test_seq)
        X_train_stats_seq, X_test_stats_seq, stats_scaler = _scale_stat_data(
            X_train_stats_seq,
            X_test_stats_seq,
        )

    return {
        "mode": "sequence_with_stats",
        "X_train": X_train_seq,
        "X_train_stats": X_train_stats_seq,
        "y_train": y_train_enc,
        "X_test": X_test_seq,
        "X_test_stats": X_test_stats_seq,
        "y_test": y_test_enc,
        "y_train_raw": y_train_seq,
        "y_test_raw": y_test_seq,
        "meta_train": meta_train_seq,
        "meta_test": meta_test_seq,
        "label_encoder": label_encoder,
        "scaler": signal_scaler,
        "stats_scaler": stats_scaler,
        "stats_feature_names": window_data["stats_feature_names"],
        "seq_len": seq_len,
        "sequence_stride": sequence_stride if sequence_stride is not None else seq_len,
        "grouping_col": grouping_col,
    }