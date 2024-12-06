import queue

class Node:
    def __init__(self, freq, key=-1, left=None, right=None, code=''):
        """
        Узел дерева Хаффмана.

        Args:
            freq: Частота символа.
            key: Символ (ключ).
            left: Левое поддерево.
            right: Правое поддерево.
            code: Код Хаффмана для этого узла.
        """
        self.freq = freq
        self.key = key
        self.left = left
        self.right = right
        self.code = code

    def __lt__(self, other):
        """Сравнение узлов по частоте для приоритетной очереди."""
        return self.freq < other.freq
    
def huffman_code(data):
    """
    Генерирует кодировку Хаффмана для заданных данных.

    Args:
        data: Входная строка данных.

    Returns:
        Словарь, где ключи — символы, а значения — их коды Хаффмана.
    """
    freq_table, code_table = {}, {}
    node_list = []
    que = queue.PriorityQueue()

    # Подсчет частоты символов
    for n in data:
        if n in freq_table:
            freq_table[n] += 1
        else:
            freq_table[n] = 1

    # Создание узлов дерева Хаффмана
    for k, v in freq_table.items():
        node_list.append(Node(v, k))
        que.put(node_list[-1])

    # Построение дерева Хаффмана
    while que.qsize() > 1:
        n1, n2 = que.get(), que.get()
        n1.code, n2.code = '1', '0'
        nn = Node(n1.freq + n2.freq, left=n1, right=n2)
        node_list.append(nn)
        que.put(node_list[-1])

    # Рекурсивное получение кодов Хаффмана
    def get_codes(node, code_str=[]):
        code_str.append(node.code)
        if node.left:
            get_codes(node.left, code_str.copy())
            get_codes(node.right, code_str.copy())
        else:
            code_table[node.key] = ''.join(code_str)

    get_codes(node_list[-1])
    return code_table