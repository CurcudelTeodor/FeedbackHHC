def accuracy(output, target, size):
    correctly_classified = 0
    for o, t in zip(output, target):
        if o.item() == t:
            correctly_classified += 1
    return correctly_classified / size


def precision(output, target, limit):
    """
    Computes precision for values in a specific interval
    :param output: Output made by the model
    :param target: The targeted labels
    :param limit: Index
    :return:
    """
    tp = 0
    fp = 0

    for o, t in zip(output, target):
        if limit == o:
            if limit == t:
                tp += 1
            else:
                fp += 1
    if tp + fp == 0:
        return 0

    return tp / (tp + fp)


def recall(output, target, limit):
    tp = 0
    fn = 0

    for o, t in zip(output, target):
        if limit == t:
            if limit == o:
                tp += 1
            else:
                fn += 1

    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def f1_score(output, target, limit):
    pr = precision(output, target, limit)
    re = recall(output, target, limit)

    if pr + re == 0:
        return 0

    return 2 * ((pr * re) / (pr + re))
