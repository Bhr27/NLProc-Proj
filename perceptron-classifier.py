import csv
import random
import re
from evaluation import Evaluator


MAX_EXAMPLES = 5000
EPOCHS = 3


def load_imdb_dataset(path, max_examples=None):
    dataset = []

    with open(path, encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            text = row["review"]
            sentiment = row["sentiment"]

            label = 1 if sentiment == "positive" else -1
            dataset.append((text, label))

    random.seed(42)
    random.shuffle(dataset)

    # We use a subset because this implementation uses plain Python lists
    # and no machine learning libraries, so full IMDB training is slow locally.
    if max_examples is not None:
        dataset = dataset[:max_examples]

    return dataset


STOP_WORDS = {
    "the", "a", "an", "is", "are", "am", "to", "for", "of", "and",
    "or", "in", "on", "at", "this", "that", "it", "its", "was", "were",
    "be", "been", "being", "with", "as", "by", "from", "but", "about",
    "into", "than", "then", "so", "if", "out", "up", "down", "very",
    "can", "could", "should", "would", "will", "just", "not", "i", "you",
    "he", "she", "they", "we", "me", "my", "your", "his", "her", "their"
}


def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"\b[a-zA-Z]+\b", text)
    tokens = [token for token in tokens if token not in STOP_WORDS]
    return tokens


def train_test_split(dataset, test_ratio=0.2):
    split_index = int(len(dataset) * (1 - test_ratio))
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]
    return train_data, test_data


def build_vocab(train_data):
    vocab = {}

    for text, _ in train_data:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab


def vectorize(text, vocab):
    vector = [0] * len(vocab)

    for token in tokenize(text):
        if token in vocab:
            vector[vocab[token]] = 1

    return vector


class PerceptronClassifier:
    def __init__(self, vocab):
        self.vocab = vocab
        self.weights = [0] * len(vocab)
        self.bias = 0

    def predict_vector(self, x):
        score = self.bias

        for i in range(len(self.weights)):
            score += self.weights[i] * x[i]

        if score > 0:
            return 1
        else:
            return -1

    def predict(self, text):
        x = vectorize(text, self.vocab)
        return self.predict_vector(x)

    def update(self, x, y):
        prediction = self.predict_vector(x)

        if prediction != y:
            for i in range(len(self.weights)):
                self.weights[i] += y * x[i]

            self.bias += y

    def train(self, train_data, epochs=5):
        for epoch in range(epochs):
            mistakes = 0

            for text, label in train_data:
                x = vectorize(text, self.vocab)
                prediction = self.predict_vector(x)

                if prediction != label:
                    self.update(x, label)
                    mistakes += 1

            print("Epoch", epoch + 1, "- mistakes:", mistakes)

dataset = load_imdb_dataset(
    "data/IMDB Dataset.csv",
    max_examples=MAX_EXAMPLES
)

train_data, test_data = train_test_split(dataset, test_ratio=0.2)

vocab = build_vocab(train_data)

classifier = PerceptronClassifier(vocab)

classifier.train(train_data, epochs=EPOCHS)

y_true = []
y_pred = []

for text, label in test_data:
    prediction = classifier.predict(text)

    y_true.append(1 if label == 1 else 0)
    y_pred.append(1 if prediction == 1 else 0)

evaluator = Evaluator(y_true, y_pred)

print()
print("Evaluation Results")
print("Accuracy:", evaluator.accuracy())
print("Precision:", evaluator.precision())
print("Recall:", evaluator.recall())
print("F1:", evaluator.f1_score())

print()
print("Example Predictions")

examples = [
    "This movie was amazing and I really loved it",
    "The film was boring and terrible",
    "The acting was great but the story was bad",
    "I would not recommend this movie"
]

for example in examples:
    prediction = classifier.predict(example)

    if prediction == 1:
        label = "positive"
    else:
        label = "negative"

    print(example, "->", label)

print()
print("Some Word Weights")

count = 0
for word, index in vocab.items():
    print(word, ":", classifier.weights[index])
    count += 1

    if count == 30:
        break