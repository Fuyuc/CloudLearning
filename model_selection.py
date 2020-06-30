import numpy as np

def train_test_split(X: object, y: object, test_ratio: object = 0.2, seed: object = None) -> object:
    """将数据 X 和 y 按照test_ratio分割成X_train, X_test, y_train, y_test（初定训练集占80%）"""

    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = np.array(X)[train_indexes]
    y_train = np.array(y)[train_indexes]

    X_test = np.array(X)[test_indexes]
    y_test = np.array(y)[test_indexes]

    return X_train, X_test, y_train, y_test
