from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.windowing import create_windows


# =========================
# METADATA / PATH HELPERS
# =========================
def parse_metadata_from_path(file_path: str, base_folder: str = "data_raw") -> dict:
    """
    Extrai metadados básicos do caminho do arquivo.

    Estruturas aceitas:
    - data_raw/Dataset/Vibration/Motor-2/50/falha/arquivo.csv
    - data_raw/Vibration/Motor-2/50/falha/arquivo.csv
    """
    normalized = Path(file_path)
    parts = list(normalized.parts)

    base_idx = None
    for i, part in enumerate(parts):
        if part == base_folder:
            base_idx = i
            break

    relative_parts = parts[base_idx + 1:] if base_idx is not None else parts

    if relative_parts and str(relative_parts[0]).lower() == "dataset":
        relative_parts = relative_parts[1:]

    file_name = normalized.name

    sensor = relative_parts[0] if len(relative_parts) > 0 else ""
    equipamento = relative_parts[1] if len(relative_parts) > 1 else ""
    rpm = relative_parts[2] if len(relative_parts) > 2 else ""
    falha = relative_parts[3] if len(relative_parts) > 3 else ""

    return {
        "sensor": str(sensor),
        "equipamento": str(equipamento),
        "rpm": str(rpm),
        "falha": str(falha),
        "canal_arquivo": extract_channel_from_filename(file_name),
        "arquivo": file_name,
        "caminho": str(normalized),
    }


def extract_channel_from_filename(file_name: str) -> str:
    stem = Path(file_name).stem.lower()

    if "-ch" in stem:
        suffix = stem.split("-ch")[-1]
        digits = "".join(c for c in suffix if c.isdigit())
        return f"ch{digits}" if digits else ""

    if "_ch" in stem:
        suffix = stem.split("_ch")[-1]
        digits = "".join(c for c in suffix if c.isdigit())
        return f"ch{digits}" if digits else ""

    return ""


# =========================
# CSV HELPERS
# =========================
def load_csv_sample(
    file_path: str,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    return pd.read_csv(file_path, nrows=nrows, low_memory=False)


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.columns.tolist()


def select_signal_columns(
    df: pd.DataFrame,
    selected_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Se selected_columns=None:
    - pega todas as colunas numéricas
    - exclui colunas típicas de tempo/índice
    """
    if selected_columns is None:
        numeric_cols = get_numeric_columns(df)

        excluded_cols = {"time", "timestamp", "index"}
        signal_cols = [
            col for col in numeric_cols
            if str(col).strip().lower() not in excluded_cols
        ]

        if not signal_cols:
            raise ValueError("Nenhuma coluna numérica de sinal encontrada no CSV.")

        return df[signal_cols].copy()

    missing = [col for col in selected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas não encontradas no CSV: {missing}")

    return df[selected_columns].copy()


def inspect_csv(
    file_path: str,
    nrows: int = 5,
) -> dict:
    df = load_csv_sample(file_path=file_path, nrows=nrows)

    return {
        "columns": df.columns.astype(str).tolist(),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
        "shape": df.shape,
        "head": df.head().to_dict(orient="records"),
    }


# =========================
# LEGACY: ARQUIVO ISOLADO
# =========================
def window_csv_file(
    file_path: str,
    selected_columns: Optional[list[str]] = None,
    window_size: int = 1024,
    overlap: float = 0.5,
    drop_last: bool = True,
    nrows: Optional[int] = None,
    base_folder: str = "data_raw",
) -> tuple[np.ndarray, dict]:
    """
    Fluxo legado por arquivo isolado.
    Mantido por compatibilidade, mas não é o fluxo correto
    para vibração consolidada por amostra.
    """
    df = load_csv_sample(file_path=file_path, nrows=nrows)
    signal_df = select_signal_columns(df, selected_columns=selected_columns)

    signal_array = signal_df.to_numpy(dtype=np.float32)

    if signal_array.ndim != 2:
        raise ValueError("O sinal convertido do CSV deveria ser 2D.")

    if signal_array.shape[1] == 1:
        signal_array = signal_array[:, 0]

    windows = create_windows(
        signal=signal_array,
        window_size=window_size,
        overlap=overlap,
        drop_last=drop_last,
    )

    metadata = parse_metadata_from_path(file_path=file_path, base_folder=base_folder)
    metadata["n_rows"] = len(df)
    metadata["n_signal_columns"] = signal_df.shape[1]
    metadata["signal_columns"] = signal_df.columns.astype(str).tolist()
    metadata["n_windows"] = len(windows)
    metadata["window_size"] = window_size
    metadata["overlap"] = overlap

    return windows, metadata


# =========================
# NOVO FLUXO: AMOSTRA CONSOLIDADA DE VIBRAÇÃO
# =========================
def _validate_vibration_sample_row(sample_row: pd.Series) -> None:
    required = ["sensor", "path_ch1", "path_ch2", "path_ch3", "path_ch4", "path_ch5"]
    missing = [col for col in required if col not in sample_row.index]
    if missing:
        raise ValueError(f"Colunas ausentes na linha da amostra: {missing}")

    sensor = str(sample_row["sensor"]).strip()
    if sensor != "Vibration":
        raise ValueError(f"Esperado sensor='Vibration', recebido '{sensor}'")

    for ch in range(1, 6):
        path_value = str(sample_row[f"path_ch{ch}"]).strip()
        if not path_value:
            raise ValueError(f"path_ch{ch} vazio na amostra.")


def _load_vibration_channel_segments(
    file_path: str,
    selected_columns: Optional[list[str]] = None,
    nrows: Optional[int] = None,
) -> tuple[np.ndarray, list[str], int]:
    """
    Lê 1 CSV de vibração (1 canal físico) e devolve:
    - array 2D: (n_samples, n_segments)
    - nomes das colunas de segmento
    - n_rows originais do csv
    """
    df = load_csv_sample(file_path=file_path, nrows=nrows)
    signal_df = select_signal_columns(df, selected_columns=selected_columns)

    arr = signal_df.to_numpy(dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Esperado array 2D no canal {file_path}, recebido {arr.shape}")

    return arr, signal_df.columns.astype(str).tolist(), len(df)


def load_vibration_sample_segments(
    sample_row: pd.Series,
    selected_columns: Optional[list[str]] = None,
    nrows: Optional[int] = None,
) -> tuple[list[np.ndarray], dict]:
    """
    Carrega uma amostra consolidada de vibração.

    Entrada:
    - sample_row com path_ch1..path_ch5

    Saída:
    - lista de segmentos, onde cada item tem shape (n_samples, 5)
    - metadata geral da amostra
    """
    _validate_vibration_sample_row(sample_row)

    channel_arrays = []
    channel_segment_names = []
    channel_n_rows = []

    channel_paths = [str(sample_row[f"path_ch{i}"]).strip() for i in range(1, 6)]

    for path in channel_paths:
        arr, seg_names, n_rows_csv = _load_vibration_channel_segments(
            file_path=path,
            selected_columns=selected_columns,
            nrows=nrows,
        )
        channel_arrays.append(arr)
        channel_segment_names.append(seg_names)
        channel_n_rows.append(n_rows_csv)

    n_segments_list = [arr.shape[1] for arr in channel_arrays]
    n_samples_list = [arr.shape[0] for arr in channel_arrays]

    if len(set(n_segments_list)) != 1:
        raise ValueError(
            f"Inconsistência no número de segmentos entre canais: {n_segments_list}"
        )

    if len(set(n_samples_list)) != 1:
        raise ValueError(
            f"Inconsistência no número de amostras por segmento entre canais: {n_samples_list}"
        )

    reference_segment_names = channel_segment_names[0]
    for i, seg_names in enumerate(channel_segment_names[1:], start=2):
        if seg_names != reference_segment_names:
            raise ValueError(
                f"Colunas/segmentos diferentes entre canais. ch1 != ch{i}"
            )

    n_segments = n_segments_list[0]
    n_samples_per_segment = n_samples_list[0]

    segment_arrays: list[np.ndarray] = []

    for seg_idx in range(n_segments):
        # empilha canais na última dimensão
        # resultado: (n_samples, 5)
        seg = np.stack(
            [channel_arrays[ch_idx][:, seg_idx] for ch_idx in range(5)],
            axis=-1,
        ).astype(np.float32)

        segment_arrays.append(seg)

    metadata = {
        "sensor": str(sample_row.get("sensor", "")),
        "equipamento": str(sample_row.get("equipamento", "")),
        "rpm": str(sample_row.get("rpm", "")),
        "falha": str(sample_row.get("falha", "")),
        "signal_id": str(sample_row.get("signal_id", "")),
        "grupo_condicao": str(sample_row.get("grupo_condicao", "")),
        "n_channels": 5,
        "n_segments": int(n_segments),
        "n_samples_per_segment": int(n_samples_per_segment),
        "segment_names": reference_segment_names,
        "channel_paths": channel_paths,
        "channel_n_rows_csv": channel_n_rows,
    }

    return segment_arrays, metadata


def window_vibration_sample(
    sample_row: pd.Series,
    window_size: int = 1024,
    overlap: float = 0.5,
    drop_last: bool = True,
    selected_columns: Optional[list[str]] = None,
    nrows: Optional[int] = None,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """
    Fluxo correto para 1 amostra de vibração consolidada.

    Passos:
    - lê ch1..ch5
    - monta segmentos (n_samples, 5)
    - faz janelamento por segmento
    - concatena todas as janelas

    Saída:
    - X_windows: (n_windows_total, window_size, 5)
    - meta_windows_df: 1 linha por janela
    - metadata geral
    """
    segment_arrays, metadata = load_vibration_sample_segments(
        sample_row=sample_row,
        selected_columns=selected_columns,
        nrows=nrows,
    )

    windows_list: list[np.ndarray] = []
    meta_rows: list[dict] = []

    for seg_idx, seg_array in enumerate(segment_arrays):
        windows = create_windows(
            signal=seg_array,
            window_size=window_size,
            overlap=overlap,
            drop_last=drop_last,
        )

        if len(windows) == 0:
            continue

        windows_list.append(windows)

        for win_idx in range(len(windows)):
            meta_rows.append({
                "sensor": metadata["sensor"],
                "equipamento": metadata["equipamento"],
                "rpm": metadata["rpm"],
                "falha": metadata["falha"],
                "signal_id": metadata["signal_id"],
                "grupo_condicao": metadata["grupo_condicao"],
                "segment_index": int(seg_idx),
                "segment_name": str(metadata["segment_names"][seg_idx]),
                "window_index": int(win_idx),
                "n_channels": 5,
                "window_size": int(window_size),
            })

    if not windows_list:
        X_windows = np.empty((0, window_size, 5), dtype=np.float32)
    else:
        X_windows = np.concatenate(windows_list, axis=0).astype(np.float32)

    meta_windows_df = pd.DataFrame(meta_rows)

    metadata["window_size"] = int(window_size)
    metadata["overlap"] = float(overlap)
    metadata["drop_last"] = bool(drop_last)
    metadata["n_total_windows"] = int(len(X_windows))

    return X_windows, meta_windows_df, metadata


def inspect_vibration_sample(
    sample_row: pd.Series,
    nrows: Optional[int] = 10,
) -> dict:
    """
    Debug rápido da amostra consolidada.
    """
    segment_arrays, metadata = load_vibration_sample_segments(
        sample_row=sample_row,
        selected_columns=None,
        nrows=nrows,
    )

    return {
        "sensor": metadata["sensor"],
        "equipamento": metadata["equipamento"],
        "rpm": metadata["rpm"],
        "falha": metadata["falha"],
        "signal_id": metadata["signal_id"],
        "grupo_condicao": metadata["grupo_condicao"],
        "n_segments": metadata["n_segments"],
        "n_samples_per_segment": metadata["n_samples_per_segment"],
        "first_segment_shape": segment_arrays[0].shape if segment_arrays else None,
        "segment_names_head": metadata["segment_names"][:5],
        "channel_paths": metadata["channel_paths"],
    }