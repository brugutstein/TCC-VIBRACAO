from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd


def validate_cv_inputs(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    n_splits: int,
) -> None:
    if group_col not in df.columns:
        raise ValueError(f"Coluna '{group_col}' não encontrada no DataFrame.")

    if label_col not in df.columns:
        raise ValueError(f"Coluna '{label_col}' não encontrada no DataFrame.")

    if n_splits < 2:
        raise ValueError("n_splits deve ser >= 2.")

    label_group_counts = df.groupby(label_col)[group_col].nunique()

    too_small = label_group_counts[label_group_counts < n_splits]
    if not too_small.empty:
        raise ValueError(
            "Algumas classes possuem menos grupos do que n_splits. "
            f"Reduza n_splits ou filtre o dataset.\n{too_small.to_dict()}"
        )


def stratified_group_kfold_split(
    df: pd.DataFrame,
    group_col: str = "grupo_condicao",
    label_col: str = "falha",
    n_splits: int = 3,
    random_state: int = 42,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Gera folds por grupo, distribuindo os grupos dentro de cada label.

    Estratégia:
    - Para cada label (falha), pega os grupos únicos daquela classe
    - Embaralha esses grupos
    - Divide os grupos dessa classe em n_splits
    - Junta os pedaços por fold

    Resultado:
    - cada fold recebe grupos de múltiplas classes
    - reduz a chance de classe inédita aparecer no teste
    - mantém integridade do grupo (sem leakage)

    Retorna:
    - lista de tuplas (train_df, test_df)
    """
    validate_cv_inputs(df, group_col, label_col, n_splits)

    rng = np.random.default_rng(random_state)

    # cada posição da lista representa os grupos de teste de um fold
    fold_test_groups = [set() for _ in range(n_splits)]

    # distribui grupos por classe
    for label, df_label in df.groupby(label_col):
        groups = df_label[group_col].dropna().unique().tolist()
        rng.shuffle(groups)

        split_groups = np.array_split(groups, n_splits)

        for fold_idx in range(n_splits):
            fold_test_groups[fold_idx].update(split_groups[fold_idx])

    all_groups = set(df[group_col].dropna().unique())
    splits = []

    for fold_idx in range(n_splits):
        test_groups = fold_test_groups[fold_idx]
        train_groups = all_groups - test_groups

        train_df = df[df[group_col].isin(train_groups)].copy()
        test_df = df[df[group_col].isin(test_groups)].copy()

        splits.append((train_df, test_df))

    return splits


def summarize_folds(
    splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
    group_col: str = "grupo_condicao",
    label_col: str = "falha",
) -> pd.DataFrame:
    """
    Gera resumo tabular dos folds.
    """
    rows = []

    for i, (train_df, test_df) in enumerate(splits):
        train_groups = set(train_df[group_col].dropna().unique())
        test_groups = set(test_df[group_col].dropna().unique())

        rows.append({
            "fold": i,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "intersection_groups": len(train_groups.intersection(test_groups)),
            "train_labels": train_df[label_col].nunique(),
            "test_labels": test_df[label_col].nunique(),
        })

    return pd.DataFrame(rows)