class Evaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = list(set(y_true + y_pred))

    def confusion_matrix(self):
        tp = fp = fn = tn = 0

        for i in range(len(self.y_true)):
            if self.y_true[i] == 1 and self.y_pred[i] == 1:
                tp += 1
            elif self.y_true[i] == 0 and self.y_pred[i] == 1:
                fp += 1
            elif self.y_true[i] == 1 and self.y_pred[i] == 0:
                fn += 1
            else:
                tn += 1

        return tp, fp, fn, tn

    def accuracy(self):
        correct = 0

        for i in range(len(self.y_true)):
            if self.y_true[i] == self.y_pred[i]:
                correct += 1

        return correct / len(self.y_true)

    def precision(self):
        tp, fp, fn, tn = self.confusion_matrix()

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    def recall(self):
        tp, fp, fn, tn = self.confusion_matrix()

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)

    def f1_score(self):
        p = self.precision()
        r = self.recall()

        if p + r == 0:
            return 0.0

        return 2 * p * r / (p + r)

    def confusion_matrix_multiclass(self):
        matrix = {}

        for true_label in self.labels:
            matrix[true_label] = {}

            for pred_label in self.labels:
                matrix[true_label][pred_label] = 0

        for i in range(len(self.y_true)):
            true = self.y_true[i]
            pred = self.y_pred[i]
            matrix[true][pred] += 1

        return matrix

    def precision_for_class(self, label):
        tp = fp = 0

        for i in range(len(self.y_true)):
            if self.y_pred[i] == label:
                if self.y_true[i] == label:
                    tp += 1
                else:
                    fp += 1

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    def recall_for_class(self, label):
        tp = fn = 0

        for i in range(len(self.y_true)):
            if self.y_true[i] == label:
                if self.y_pred[i] == label:
                    tp += 1
                else:
                    fn += 1

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)

    def f1_for_class(self, label):
        p = self.precision_for_class(label)
        r = self.recall_for_class(label)

        if p + r == 0:
            return 0.0

        return 2 * p * r / (p + r)

    def macro_precision(self):
        total = 0

        for label in self.labels:
            total += self.precision_for_class(label)

        return total / len(self.labels)

    def macro_recall(self):
        total = 0

        for label in self.labels:
            total += self.recall_for_class(label)

        return total / len(self.labels)

    def macro_f1(self):
        total = 0

        for label in self.labels:
            total += self.f1_for_class(label)

        return total / len(self.labels)


def run_tests():
    test_cases = [
        ([1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 1, 1]),
        ([1, 0, 1, 0], [0, 0, 0, 0]),
        ([0, 0, 0, 0], [1, 0, 1, 0]),
    ]

    for y_true, y_pred in test_cases:
        evaluator = Evaluator(y_true, y_pred)

        print("y_true:", y_true)
        print("y_pred:", y_pred)
        print("Confusion Matrix:", evaluator.confusion_matrix())
        print("Accuracy:", evaluator.accuracy())
        print("Precision:", evaluator.precision())
        print("Recall:", evaluator.recall())
        print("F1:", evaluator.f1_score())
        print("--------------------")

    custom_test_cases = [
        ([1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]),
        ([1, 1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]),
    ]

    for y_true, y_pred in custom_test_cases:
        evaluator = Evaluator(y_true, y_pred)

        print("Custom test")
        print("y_true:", y_true)
        print("y_pred:", y_pred)
        print("Confusion Matrix:", evaluator.confusion_matrix())
        print("Accuracy:", evaluator.accuracy())
        print("Precision:", evaluator.precision())
        print("Recall:", evaluator.recall())
        print("F1:", evaluator.f1_score())
        print("--------------------")

    multi = Evaluator(
        ["positive", "negative", "neutral", "positive"],
        ["positive", "positive", "neutral", "negative"]
    )

    print("Multi-class test")
    print("Confusion Matrix:", multi.confusion_matrix_multiclass())
    print("Accuracy:", multi.accuracy())
    print("Macro Precision:", multi.macro_precision())
    print("Macro Recall:", multi.macro_recall())
    print("Macro F1:", multi.macro_f1())


if __name__ == "__main__":
    run_tests()