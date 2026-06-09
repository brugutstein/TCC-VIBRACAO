from __future__ import annotations

from pathlib import Path
import os
import re

import pandas as pd

from src.ingest import parse_metadata_from_path


EXPECTED_CHANNELS = {
    "Vibration": 5,
    "Electric": 6,
}


def _safe_int(value):
    try:
        return int(value)
    except Exception:
        return value


def _read_csv_header_only(file_path: str) -> list[str]:
    """
    Lê só o cabeçalho do CSV.
    """
    df = pd.read_csv(file_path, nrows=0, low_memory=False)
    return [str(c) for c in df.columns.tolist()]


def _normalize_channel_label(channel_value: str) -> str:
    """
    Normaliza canal para padrão ch1, ch2, ...
    """
    s = str(channel_value).strip().lower()

    if not s:
        return ""

    if s.startswith("ch"):
        digits = "".join(c for c in s if c.isdigit())
        return f"ch{digits}" if digits else s

    if s.isdigit():
        return f"ch{s}"

    match = re.search(r"(\d+)$", s)
    if match:
        return f"ch{match.group(1)}"

    return s


def extract_base_signal_id(file_name: str) -> str:
    """
    Remove o sufixo do canal para agrupar arquivos da mesma medição.
    Ex:
    - xxx-ch1.csv -> xxx
    - xxx_ch2.csv -> xxx
    """
    stem = Path(file_name).stem.lower()

    if "-ch" in stem:
        return stem.split("-ch")[0]

    if "_ch" in stem:
        return stem.split("_ch")[0]

    return stem


def scan_dataset_raw(
    base_path: str,
    base_folder: str = "data_raw",
    inspect_headers: bool = True,
) -> pd.DataFrame:
    """
    Varre todos os CSVs e monta catálogo bruto por arquivo.
    """
    rows = []

    for root, _, files in os.walk(base_path):
        for file_name in files:
            if not file_name.lower().endswith(".csv"):
                continue

            full_path = str(Path(root) / file_name)
            file_size_mb = Path(full_path).stat().st_size / (1024 ** 2)

            metadata = parse_metadata_from_path(
                file_path=full_path,
                base_folder=base_folder,
            )

            header_cols = []
            n_header_columns = None

            if inspect_headers:
                try:
                    header_cols = _read_csv_header_only(full_path)
                    n_header_columns = len(header_cols)
                except Exception:
                    header_cols = []
                    n_header_columns = None

            sensor = str(metadata.get("sensor", "")).strip()
            equipamento = str(metadata.get("equipamento", "")).strip()
            rpm = metadata.get("rpm", "")
            falha = str(metadata.get("falha", "")).strip()
            canal_arquivo = _normalize_channel_label(metadata.get("canal_arquivo", ""))

            signal_id = extract_base_signal_id(file_name)

            grupo_condicao = f"{sensor}__{equipamento}__{rpm}__{falha}__{signal_id}"

            row = {
                "sensor": sensor,
                "equipamento": equipamento,
                "rpm": _safe_int(rpm),
                "falha": falha,
                "signal_id": signal_id,
                "grupo_condicao": grupo_condicao,
                "canal_arquivo": canal_arquivo,
                "arquivo": metadata.get("arquivo", ""),
                "caminho": metadata.get("caminho", ""),
                "tamanho_mb": round(file_size_mb, 2),
                "n_header_columns": n_header_columns,
                "header_columns": "|".join(header_cols) if header_cols else "",
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        sort_cols = [
            c for c in [
                "sensor",
                "equipamento",
                "rpm",
                "falha",
                "signal_id",
                "canal_arquivo",
                "arquivo",
            ] if c in df.columns
        ]
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def build_sample_catalog(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Consolida o catálogo em nível de amostra física.

    Para vibração:
    - 1 linha = 1 condição física / medição
    - paths ch1..ch5 explícitos
    - valida se os 5 canais existem
    - valida consistência de número de colunas entre canais

    Para elétrico:
    - mesmo raciocínio, mas espera 6 canais
    """
    required_cols = {
        "sensor",
        "equipamento",
        "rpm",
        "falha",
        "signal_id",
        "grupo_condicao",
        "canal_arquivo",
        "arquivo",
        "caminho",
        "n_header_columns",
    }
    missing = required_cols - set(raw_df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no raw_df: {missing}")

    rows = []

    for group_key, g in raw_df.groupby("grupo_condicao", sort=False):
        g = g.copy().reset_index(drop=True)

        sensor = str(g["sensor"].iloc[0]).strip()
        equipamento = str(g["equipamento"].iloc[0]).strip()
        rpm = g["rpm"].iloc[0]
        falha = str(g["falha"].iloc[0]).strip()
        signal_id = str(g["signal_id"].iloc[0]).strip()

        expected_channels = EXPECTED_CHANNELS.get(sensor, None)

        channel_map = {}
        channel_file_map = {}
        ncols_map = {}

        for _, row in g.iterrows():
            ch = _normalize_channel_label(row["canal_arquivo"])
            if not ch:
                continue

            channel_map[ch] = row["caminho"]
            channel_file_map[ch] = row["arquivo"]
            ncols_map[ch] = row["n_header_columns"]

        found_channels = sorted(
            channel_map.keys(),
            key=lambda x: int(re.sub(r"\D", "", x) or 999)
        )

        n_channels_found = len(found_channels)
        is_channel_count_ok = (
            expected_channels is None or n_channels_found == expected_channels
        )

        valid_ncols = [v for v in ncols_map.values() if pd.notna(v)]
        unique_ncols = sorted(set(valid_ncols))
        same_segment_count = len(unique_ncols) <= 1
        n_segments = unique_ncols[0] if len(unique_ncols) == 1 else None

        row_out = {
            "sensor": sensor,
            "equipamento": equipamento,
            "rpm": rpm,
            "falha": falha,
            "signal_id": signal_id,
            "grupo_condicao": group_key,
            "expected_n_channels": expected_channels,
            "n_channels_found": n_channels_found,
            "found_channels": "|".join(found_channels),
            "is_channel_count_ok": bool(is_channel_count_ok),
            "same_segment_count_across_channels": bool(same_segment_count),
            "n_segments_per_channel": n_segments,
            "is_sample_valid": bool(is_channel_count_ok and same_segment_count),
        }

        max_channels = expected_channels if expected_channels is not None else max(n_channels_found, 1)

        for i in range(1, max_channels + 1):
            ch = f"ch{i}"
            row_out[f"path_{ch}"] = channel_map.get(ch, "")
            row_out[f"file_{ch}"] = channel_file_map.get(ch, "")
            row_out[f"ncols_{ch}"] = ncols_map.get(ch, None)

        rows.append(row_out)

    sample_df = pd.DataFrame(rows)

    if not sample_df.empty:
        sample_df = sample_df.sort_values(
            ["sensor", "equipamento", "rpm", "falha", "signal_id"]
        ).reset_index(drop=True)

    return sample_df


def save_catalog(df: pd.DataFrame, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() == ".csv":
        df.to_csv(output, index=False)
    elif output.suffix.lower() == ".parquet":
        df.to_parquet(output, index=False)
    else:
        raise ValueError("Use .csv ou .parquet")


if __name__ == "__main__":
    base_path = "data_raw"

    raw_df = scan_dataset_raw(
        base_path=base_path,
        base_folder="data_raw",
        inspect_headers=True,
    )
    
    print("\n[DEBUG] colunas raw:")
    print(raw_df.columns.tolist())

    print("\n[DEBUG] arquivos / canal / signal_id:")
    print(raw_df[["arquivo", "canal_arquivo", "signal_id", "grupo_condicao"]].head(20).to_string())

    print("\n[DEBUG] canais únicos:")
    print(raw_df["canal_arquivo"].value_counts(dropna=False).head(20))

    print("\n[DEBUG] signal_id únicos exemplo:")
    print(raw_df["signal_id"].head(20).to_string(index=False))
        
    save_catalog(raw_df, "outputs/catalog_raw_files.csv")

    sample_df = build_sample_catalog(raw_df)
    save_catalog(sample_df, "outputs/catalog_samples.csv")

    print("\n[RAW] shape:", raw_df.shape)
    print("\n[RAW] canais por grupo:")
    print(raw_df.groupby("grupo_condicao")["canal_arquivo"].nunique().value_counts())

    print("\n[SAMPLE] shape:", sample_df.shape)
    print("\n[SAMPLE] validade:")
    print(sample_df["is_sample_valid"].value_counts(dropna=False))
    print("\n[SAMPLE] canais encontrados:")
    print(sample_df["n_channels_found"].value_counts(dropna=False))
    print("\n[SAMPLE] segmentos consistentes:")
    print(sample_df["same_segment_count_across_channels"].value_counts(dropna=False))

    print("\nCatálogo bruto salvo em: outputs/catalog_raw_files.csv")
    print("Catálogo consolidado salvo em: outputs/catalog_samples.csv")