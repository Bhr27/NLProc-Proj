def confusion_matrix(y_true, y_pred):
    tp = fp = fn = tn = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1 
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
        else:
            tn += 1
        
    return tp, fp, fn, tn


y_true = [1, 1, 0, 0]
y_pred = [0, 0, 1, 1]

def accuracy(y_true, y_pred):
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred)

    correct = tp + tn
    total = len(y_true)

    return correct / total

print("accuracy:" , accuracy(y_true, y_pred))

def precision(y_true, y_pred):
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred)

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)

print("Precision:", precision(y_true, y_pred))

def recall(y_true, y_pred):
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred)

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)

print("Recall:", recall(y_true, y_pred))


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)

print("F1 Score:", f1_score(y_true, y_pred))





