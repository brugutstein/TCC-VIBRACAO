from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# =========================================================
# BUILD MLP CATALOG
# Lê o catalog_samples.csv existente, filtra as classes alvo
# e salva em outputs/catalog_samples.csv (onde o cache espera).
# =========================================================

PROJECT_ROOT = Path(r"F:\TCC\Dataset\NLM-TCC\pump_fault_tcc")

# catalog_samples.csv gerado anteriormente (fonte da verdade)
CATALOG_SOURCE = PROJECT_ROOT / "NLN_EMP_vibration_motor2" / "catalog_samples.csv"

# onde o build_mlp_cache.py vai buscar
CATALOG_OUT = PROJECT_ROOT / "outputs" / "catalog_samples.csv"
CATALOG_OUT.parent.mkdir(parents=True, exist_ok=True)

# pasta raiz dos dados brutos (para resolver caminhos relativos)
DATA_RAW_ROOT = PROJECT_ROOT / "CNN"

TARGET_LABELS = [
    "bearing pump",
    "broken rotor bar",
    "healthy",
    "impeller",
    "loose foot motor",
    "loose foot pump",
]

LABEL_MERGE = {
    "healthy noise": "healthy",
    "healthy 1": "healthy",
    "healthy 2": "healthy",
    "healthy 3": "healthy",
}

N_CHANNELS = 5


def normalize_fault_label(label: str) -> str:
    label = str(label).lower().strip()
    label = label.replace("_", " ")
    label = label.replace("-", " ")
    label = re.sub(r"\s+", " ", label)
    # merge explícito primeiro
    if label in LABEL_MERGE:
        return LABEL_MERGE[label]
    # remove sufixo numérico: bearing pump 1 -> bearing pump
    label = re.sub(r"\s+\d+$", "", label)
    return LABEL_MERGE.get(label, label)


def resolve_channel_paths(df: pd.DataFrame) -> pd.DataFrame:
    path_cols = [f"path_ch{i}" for i in range(1, N_CHANNELS + 1)]

    def resolve_one(p):
        if pd.isna(p):
            return p

        raw = str(p).strip()
        raw_path = Path(raw)

        candidates = [
            raw_path,
            PROJECT_ROOT / raw_path,
            DATA_RAW_ROOT / raw_path,
        ]

        for c in candidates:
            if c.exists():
                return str(c)

        raise FileNotFoundError(
            "Arquivo não encontrado. Tentativas:\n"
            + "\n".join(str(c) for c in candidates)
        )

    for col in path_cols:
        df[col] = df[col].apply(resolve_one)

    return df


def main():
    print("=" * 70)
    print("BUILD MLP CATALOG")
    print("=" * 70)
    print("FONTE:", CATALOG_SOURCE)
    print("SAÍDA:", CATALOG_OUT)
    print("=" * 70)

    if not CATALOG_SOURCE.exists():
        raise FileNotFoundError(
            f"catalog_samples.csv não encontrado em:\n  {CATALOG_SOURCE}"
        )

    df = pd.read_csv(CATALOG_SOURCE)

    print(f"\nLinhas carregadas: {len(df)}")
    print(f"Colunas: {list(df.columns)}")

    # filtra sensor e equipamento
    df = df[
        (df["sensor"].astype(str) == "Vibration")
        & (df["equipamento"].astype(str) == "Motor-2")
    ].copy()

    print(f"\nApós filtro vibration/Motor-2: {len(df)} amostras")

    # normaliza label
    df["falha_base"] = df["falha"].apply(normalize_fault_label)

    # filtra classes alvo
    df = df[df["falha_base"].isin(TARGET_LABELS)].copy()

    print(f"Após filtro de classes: {len(df)} amostras")

    if df.empty:
        raise ValueError(
            "Catálogo vazio após filtros.\n"
            f"Classes disponíveis: {df['falha'].unique().tolist()}"
        )

    # filtra amostras válidas com todos os canais
    df = df[
        (df["is_sample_valid"] == True)
        & (df["n_channels_found"] == N_CHANNELS)
    ].copy()

    print(f"Após filtro is_sample_valid + {N_CHANNELS} canais: {len(df)} amostras")

    if df.empty:
        raise ValueError("Nenhuma amostra válida após filtros de qualidade.")

    # resolve caminhos relativos -> absolutos
    df = resolve_channel_paths(df)

    # valida que os arquivos existem (amostra dos primeiros)
    sample_check = df.head(3)
    for _, row in sample_check.iterrows():
        p = Path(row["path_ch1"])
        if not p.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado após resolução de caminho:\n  {p}\n"
                f"Verifique DATA_RAW_ROOT: {DATA_RAW_ROOT}"
            )

    df = df.reset_index(drop=True)

    # salva
    df.to_csv(CATALOG_OUT, index=False)

    print("\n[DISTRIBUIÇÃO POR CLASSE]")
    print(df["falha_base"].value_counts().sort_index())

    print("\n[GRUPOS POR CLASSE]")
    print(df.groupby("falha_base")["grupo_condicao"].nunique().sort_index())

    print("\n[RPMs]")
    print(df["rpm"].value_counts().sort_index())

    print(f"\nTotal de amostras catalogadas: {len(df)}")
    print(f"Classes: {sorted(df['falha_base'].unique())}")
    print(f"\nCatálogo salvo em:\n  {CATALOG_OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
