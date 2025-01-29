import numpy as np
from scipy.linalg import hankel

def binary_to_char(row, char_set="0123456789abcdefghijklmnopqrstuvwxyz"):
    """
    Преобразует бинарный список (строку из 0 и 1) в символ из заданного набора.
    Использует хеширование с остатком от деления.
    
    row: список из 0 и 1
    char_set: набор символов для использования при кодировании
    """

    binary_string = ''.join(map(str, row))
    int_value = int(binary_string, 2) 
    char_index = int_value % len(char_set) 
    return char_set[char_index]

def binary_to_hex(row):
    """ Переводит бинарный список в символ 16-ричного алфавита. """
    binary_string = ''.join(map(str, row))
    decimal_value = int(binary_string, 2)
    return hex(decimal_value)[2:]

def encode_row(row):
    """ Кодирует строку в десятичное числовое представление. """
    s =  ' '.join(map(str, row))
    mapping = {str(-2 + i): i for i in range(10)}
    decimal_value = 0
    parts = s.split()  
    for i, part in enumerate(parts):
        decimal_value += mapping[part] * (10**(len(parts) - i - 1))  
    return decimal_value


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


def calculate_f1_accuracy(x_pred, x_real):
    """Вычисляет F1-меру и точность.

    Args:
        y_pred: Предсказанные бинарные метки (массив NumPy).
        y_true: Истинные бинарные метки (массив NumPy).

    Returns:
        Кортеж: (F1-мера, точность). Возвращает (0, 0), если нет положительных предсказаний.
    """
    x_pred, x_real = x_pred.astype(int), x_real.astype(int) 
    tp = len(np.where(x_pred[np.where(x_real == 1)] == 1)[0])
    tn = len(np.where(x_pred[np.where(x_real == 0)] == 0)[0])
    fp = len(np.where(x_pred[np.where(x_real == 0)] == 1)[0])
    fn = len(np.where(x_pred[np.where(x_real == 1)] == 0)[0])
    if (tp + fp) * (tp + fn) * tp:
        precision, recall = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) 
    elif sum(x_pred - x_real):
        f1 = 0.
    else:
        f1 = 1.
    if (tp + tn + fp + fn):
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    else:
        accuracy = 0.
    return f1, accuracy
