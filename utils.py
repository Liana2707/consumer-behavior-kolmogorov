import numpy as np
from scipy.linalg import hankel


def binary_to_hex(row):
    binary_string = ''.join(map(str, row))
    decimal_value = int(binary_string, 2)
    return hex(decimal_value)[2:]


def create_dataset(time_series, lzw_depth, forecast_horizon):
    """
    Создает обучающий набор данных для прогнозирования временных рядов.

    Args:
        time_series: Временной ряд (numpy array).
        lzw_depth: Глубина Lempel-Ziv кодирования (int).
        forecast_horizon: Горизонт прогнозирования (int).

    Returns:
        Кортеж (X, y):  X - входные данные, y - целевые значения.
    """
    hankel_matrix = hankel(time_series)
    X0 = hankel_matrix[:-lzw_depth - forecast_horizon + 1, :lzw_depth]
    X = []
    for i in range(X0.shape[0] - forecast_horizon - 1):
        X.append(X0[i:i + forecast_horizon + 1, :].T)
    X = np.array(X)
    y = hankel_matrix[:-lzw_depth - 2 * forecast_horizon, lzw_depth + forecast_horizon:lzw_depth + 2 * forecast_horizon]
    return X, y


def calculate_f1_accuracy(y_pred, y_true):
    """Вычисляет F1-меру и точность.

    Args:
        y_pred: Предсказанные бинарные метки (массив NumPy).
        y_true: Истинные бинарные метки (массив NumPy).

    Returns:
        Кортеж: (F1-мера, точность). Возвращает (0, 0), если нет положительных предсказаний.
    """
    y_pred = y_pred.astype(int)
    y_true = y_true.astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    if tp + fp == 0 or tp + fn == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0

    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100 if (tp + tn + fp + fn) != 0 else 0.0

    return f1, accuracy
