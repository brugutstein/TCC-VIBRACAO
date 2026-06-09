from __future__ import annotations

import numpy as np


def validate_window_params(
    window_size: int,
    step_size: int,
) -> None:
    """
    Valida os parâmetros de janelamento.
    """
    if window_size <= 0:
        raise ValueError("window_size deve ser > 0")

    if step_size <= 0:
        raise ValueError("step_size deve ser > 0")

    if step_size > window_size:
        raise ValueError("step_size não pode ser maior que window_size")


def calculate_step_size(window_size: int, overlap: float) -> int:
    """
    Calcula o step_size com base no overlap.

    overlap:
        0.0 -> sem sobreposição
        0.5 -> 50% de sobreposição
        0.75 -> 75% de sobreposição
    """
    if not 0 <= overlap < 1:
        raise ValueError("overlap deve estar entre 0 e 1 (ex: 0.5 para 50%)")

    step_size = int(window_size * (1 - overlap))

    if step_size <= 0:
        raise ValueError("overlap muito alto gerou step_size inválido")

    return step_size


def create_windows(
    signal: np.ndarray,
    window_size: int = 1024,
    overlap: float = 0.5,
    drop_last: bool = True,
) -> np.ndarray:
    """
    Cria janelas a partir de um sinal temporal.

    Parâmetros
    ----------
    signal : np.ndarray
        Pode ser:
        - 1D: (n_samples,)
        - 2D: (n_samples, n_channels)

    window_size : int
        Tamanho da janela.

    overlap : float
        Sobreposição entre janelas.
        Exemplo:
        - 0.0 = sem overlap
        - 0.5 = 50% de overlap

    drop_last : bool
        Se True, descarta resto que não completar uma janela.
        Se False, cria padding com zeros na última.

    Retorno
    -------
    np.ndarray
        Se entrada 1D:
            (n_windows, window_size)

        Se entrada 2D:
            (n_windows, window_size, n_channels)
    """
    if not isinstance(signal, np.ndarray):
        signal = np.asarray(signal)

    if signal.ndim not in (1, 2):
        raise ValueError("signal deve ser 1D ou 2D")

    step_size = calculate_step_size(window_size, overlap)
    validate_window_params(window_size, step_size)

    n_samples = signal.shape[0]

    if n_samples < window_size:
        if drop_last:
            if signal.ndim == 1:
                return np.empty((0, window_size), dtype=signal.dtype)
            return np.empty((0, window_size, signal.shape[1]), dtype=signal.dtype)

        if signal.ndim == 1:
            padded = np.zeros(window_size, dtype=signal.dtype)
            padded[:n_samples] = signal
            return padded[np.newaxis, :]

        padded = np.zeros((window_size, signal.shape[1]), dtype=signal.dtype)
        padded[:n_samples, :] = signal
        return padded[np.newaxis, :, :]

    windows = []

    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        windows.append(signal[start:end])

    remainder = (n_samples - window_size) % step_size

    if not drop_last and remainder != 0:
        last_start = len(windows) * step_size
        tail = signal[last_start:]

        if signal.ndim == 1:
            padded = np.zeros(window_size, dtype=signal.dtype)
            padded[: len(tail)] = tail
        else:
            padded = np.zeros((window_size, signal.shape[1]), dtype=signal.dtype)
            padded[: len(tail), :] = tail

        windows.append(padded)

    return np.stack(windows, axis=0)