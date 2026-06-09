from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def validate_split_sizes(
    train_size: float,
    val_size: float,
    test_size: float,
) -> None:
    total = train_size + val_size + test_size

    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_size + val_size + test_size deve somar 1.0. Atual: {total}"
        )

    for name, value in {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }.items():
        if value <= 0 or value >= 1:
            raise ValueError(f"{name} deve estar entre 0 e 1.")


def split_groups(
    df: pd.DataFrame,
    group_col: str = "grupo_condicao",
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Faz split por grupos únicos, evitando leakage entre canais da mesma condição.
    """
    validate_split_sizes(train_size, val_size, test_size)

    if group_col not in df.columns:
        raise ValueError(f"Coluna '{group_col}' não encontrada no DataFrame.")

    unique_groups = df[group_col].dropna().unique()
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)

    n_train = int(n_groups * train_size)
    n_val = int(n_groups * val_size)

    train_groups = unique_groups[:n_train]
    val_groups = unique_groups[n_train:n_train + n_val]
    test_groups = unique_groups[n_train + n_val:]

    train_df = df[df[group_col].isin(train_groups)].copy()
    val_df = df[df[group_col].isin(val_groups)].copy()
    test_df = df[df[group_col].isin(test_groups)].copy()

    return train_df, val_df, test_df


def summarize_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str = "grupo_condicao",
    label_col: str = "falha",
) -> dict:
    """
    Retorna um resumo útil do split.
    """
    return {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_groups": train_df[group_col].nunique(),
        "val_groups": val_df[group_col].nunique(),
        "test_groups": test_df[group_col].nunique(),
        "train_labels": train_df[label_col].nunique() if label_col in train_df.columns else None,
        "val_labels": val_df[label_col].nunique() if label_col in val_df.columns else None,
        "test_labels": test_df[label_col].nunique() if label_col in test_df.columns else None,
    }



def stratified_group_split(
    df: pd.DataFrame,
    group_col: str = "grupo_condicao",
    label_col: str = "falha",
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    """
    Split por grupo + estratificação por label (falha).
    """

    rng = np.random.default_rng(random_state)

    train_groups = []
    val_groups = []
    test_groups = []

    for label, df_label in df.groupby(label_col):
        groups = df_label[group_col].unique()
        rng.shuffle(groups)

        n = len(groups)

        n_train = int(n * train_size)
        n_val = int(n * val_size)

        train_groups.extend(groups[:n_train])
        val_groups.extend(groups[n_train:n_train + n_val])
        test_groups.extend(groups[n_train + n_val:])

    train_df = df[df[group_col].isin(train_groups)].copy()
    val_df = df[df[group_col].isin(val_groups)].copy()
    test_df = df[df[group_col].isin(test_groups)].copy()

    return train_df, val_df, test_df