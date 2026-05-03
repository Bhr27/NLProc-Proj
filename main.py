import csv
import random
from evaluation import Evaluator


def load_imdb_dataset(path, limit=2000):
    dataset = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            text = row["review"]
            label = 1 if row["sentiment"] == "positive" else 0
            dataset.append((text, label))

    random.seed(42)
    random.shuffle(dataset)

    return dataset[:limit]


def tokenize(text):
    return text.lower().split()


def build_vocab(dataset):
    vocab = set()

    for text, _ in dataset:
        tokens = tokenize(text)
        vocab.update(tokens)

    return vocab


def train(train_data):
    vocab = build_vocab(train_data)

    total_words = {1: 0, 0: 0}
    class_counts = {1: 0, 0: 0}
    word_counts = {1: {}, 0: {}}

    for text, label in train_data:
        class_counts[label] += 1
        tokens = tokenize(text)

        for token in tokens:
            total_words[label] += 1
            word_counts[label][token] = word_counts[label].get(token, 0) + 1

    total_docs = len(train_data)
    priors = {}

    for label in class_counts:
        priors[label] = class_counts[label] / total_docs

    V = len(vocab)
    likelihoods = {1: {}, 0: {}}

    for label in [1, 0]:
        for word in vocab:
            word_count = word_counts[label].get(word, 0)
            likelihoods[label][word] = (word_count + 1) / (total_words[label] + V)

    return vocab, priors, likelihoods, total_words


def score(text, label, vocab, priors, likelihoods, total_words):
    tokens = tokenize(text)
    prob = priors[label]
    V = len(vocab)

    for token in tokens:
        if token in vocab:
            prob *= likelihoods[label][token]
        else:
            prob *= 1 / (total_words[label] + V)

    return prob


def predict(text, vocab, priors, likelihoods, total_words):
    scores = {}

    for label in [1, 0]:
        scores[label] = score(text, label, vocab, priors, likelihoods, total_words)

    return max(scores, key=scores.get)


dataset = load_imdb_dataset("data/IMDB Dataset.csv", limit=2000)

split = int(len(dataset) * 0.8)
train_data = dataset[:split]
test_data = dataset[split:]

vocab, priors, likelihoods, total_words = train(train_data)

y_true = []
y_pred = []

for text, label in test_data:
    prediction = predict(text, vocab, priors, likelihoods, total_words)
    y_true.append(label)
    y_pred.append(prediction)

evaluator = Evaluator(y_true, y_pred)

print("Accuracy:", evaluator.accuracy())
print("Precision:", evaluator.precision())
print("Recall:", evaluator.recall())
print("F1:", evaluator.f1_score())

