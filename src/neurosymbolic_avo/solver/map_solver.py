import numpy as np
from scipy.linalg import cholesky, cho_solve
from typing import Optional, Tuple


def solve_map(
    kernel_matrix: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    noise_level: float = 1e-6,
    max_retries: int = 3,
    noise_increment: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    K = kernel_matrix.copy()
    K_y = K + noise_level * np.eye(K.shape[0])

    for attempt in range(max_retries):
        try:
            L = cholesky(K_y, lower=True)
            alpha = cho_solve((L, True), y)
            f_map = K @ alpha
            return f_map, L
        except np.linalg.LinAlgError:
            if attempt < max_retries - 1:
                noise_level *= noise_increment
                K_y = K + noise_level * np.eye(K.shape[0])
            else:
                raise RuntimeError(f"Failed to solve MAP after {max_retries} attempts")


def solve_map_with_masking(
    kernel_matrix: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    outlier_indices: Optional[list] = None,
    noise_level: float = 1e-6,
    max_retries: int = 3,
    noise_increment: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    if outlier_indices is None or len(outlier_indices) == 0:
        return solve_map(kernel_matrix, X, y, noise_level, max_retries, noise_increment)

    valid_indices = [i for i in range(len(y)) if i not in outlier_indices]
    y_valid = y[valid_indices]
    K_valid = kernel_matrix[np.ix_(valid_indices, valid_indices)]

    f_map_valid, L = solve_map(
        K_valid,
        X[valid_indices],
        y_valid,
        noise_level,
        max_retries,
        noise_increment
    )

    f_map = np.zeros_like(y)
    f_map[valid_indices] = f_map_valid

    return f_map, L


def compute_semblance(
    y: np.ndarray,
    f_map: np.ndarray
) -> float:
    n_traces = len(y)
    
    if n_traces == 0:
        return 0.0
    
    sum_y = np.sum(y)
    sum_y2 = np.sum(y**2)
    
    if sum_y2 < 1e-10:
        return 0.0
    
    semblance = (sum_y**2) / (n_traces * sum_y2)
    return np.clip(semblance, 0.0, 1.0)


def compute_log_marginal_likelihood(
    kernel_matrix: np.ndarray,
    y: np.ndarray,
    noise_level: float = 1e-6
) -> float:
    n = len(y)
    K_y = kernel_matrix + noise_level * np.eye(n)

    try:
        L = cholesky(K_y, lower=True)
        alpha = cho_solve((L, True), y)

        log_likelihood = -0.5 * y @ alpha
        log_likelihood -= np.sum(np.log(np.diag(L)))
        log_likelihood -= 0.5 * n * np.log(2 * np.pi)

        return log_likelihood
    except np.linalg.LinAlgError:
        return -np.inf
