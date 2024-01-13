from config import DELTA


def accuracy(output, target, size):
    correctly_classified = 0
    for o, t in zip(output, target):
        if abs(o - t) < DELTA:
            correctly_classified += 1
    return correctly_classified / size


def precision(output, target, limit1, limit2):
    """
    Computes precision for values in a specific interval
    :param output: Output made by the model
    :param target: The targeted labels
    :param limit1: inferior limit of the interval
    :param limit2: superior limit of the interval
    :return:
    """
    tp = 0
    fp = 0

    for o, t in zip(output, target):
        if limit1 < o < limit2:
            if limit1 < t < limit2:
                tp += 1
            else:
                fp += 1
    if tp + fp == 0:
        return 0

    return tp / (tp + fp)


def recall(output, target, limit1, limit2):
    tp = 0
    fn = 0

    for o, t in zip(output, target):
        if limit1 < t < limit2:
            if limit1 < o < limit2:
                tp += 1
            else:
                fn += 1

    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def f1_score(output, target, limit1, limit2):
    pr = precision(output, target, limit1, limit2)
    re = recall(output, target, limit1, limit2)

    if pr + re == 0:
        return 0

    return 2 * ((pr * re) / (pr + re))
