

def lempel_ziv_complexity(text):
    """
    Вычисляет сложность Лемпеля-Зива строки.

    Сложность Лемпеля-Зива — это мера сложности строки, 
    определяемая как количество различных подстрок, необходимых для представления строки.

    Args:
        text: Входная строка.

    Returns:
        Сложность Лемпеля-Зива строки.
    """
    n = len(text)
    i = 0
    C = u = v = vmax = 1
    while (u + v) < n:
        if text[i + v] == text[u + v]:
            v += 1
        else:
            vmax = max(v, vmax)
            i += 1
            v = 1
            if i == u:
                C += 1
                u += vmax
                i, vmax = 0, v
    if v != 1:
        C += 1
    return C

def lzw_compress(text, dict_size=256):
    """
    Сжимает текст используя алгоритм LZW (Lempel-Ziv-Welch).

    Args:
        text: Входной текст (строка).
        dict_size: Начальный размер словаря (по умолчанию 256).

    Returns:
        Кортеж, содержащий:
            - Список сжатых данных (индексы из словаря).
            - Словарь, используемый для сжатия.
    """
    dictionary = {chr(i): i for i in range(dict_size)}
    compressed_data = []
    string = ""
    for symbol in text:
        new_string = string + symbol
        if new_string in dictionary:
            string = new_string
        else:
            compressed_data.append(dictionary[string])
            dictionary[new_string] = dict_size
            dict_size += 1
            string = symbol
    if string: 
        compressed_data.append(dictionary[string])
    return compressed_data, dictionary
