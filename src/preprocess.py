from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


def reshape_3d_to_2d(X: np.ndarray) -> Tuple[np.ndarray, tuple]:
    """
    Converte array 3D (n_samples, timesteps, n_features)
    para 2D (n_samples * timesteps, n_features),
    preservando o shape original para reconstrução.
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if X.ndim != 3:
        raise ValueError(
            f"Esperado array 3D (n_samples, timesteps, n_features), recebido shape {X.shape}"
        )

    original_shape = X.shape
    X_2d = X.reshape(-1, X.shape[-1])

    return X_2d, original_shape


def reshape_2d_to_3d(X_2d: np.ndarray, original_shape: tuple) -> np.ndarray:
    """
    Reconstrói array 2D para o shape 3D original.
    """
    return X_2d.reshape(original_shape)


def fit_standard_scaler(X_train: np.ndarray) -> StandardScaler:
    """
    Ajusta StandardScaler usando apenas o conjunto de treino.
    """
    X_train_2d, _ = reshape_3d_to_2d(X_train)

    scaler = StandardScaler()
    scaler.fit(X_train_2d)

    return scaler


def transform_with_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Aplica scaler em array 3D e devolve no mesmo shape original.
    """
    X_2d, original_shape = reshape_3d_to_2d(X)
    X_scaled_2d = scaler.transform(X_2d)
    X_scaled = reshape_2d_to_3d(X_scaled_2d, original_shape)

    return X_scaled.astype(np.float32)


def fit_transform_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Ajusta scaler no treino e transforma treino e teste.
    """
    scaler = fit_standard_scaler(X_train)

    X_train_scaled = transform_with_scaler(X_train, scaler)
    X_test_scaled = transform_with_scaler(X_test, scaler)

    return X_train_scaled, X_test_scaled, scaler


def save_scaler(scaler: StandardScaler, output_path: str) -> None:
    """
    Salva scaler em disco.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output)


def load_scaler(input_path: str) -> StandardScaler:
    """
    Carrega scaler salvo.
    """
    return joblib.load(input_path)