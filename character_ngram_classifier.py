import csv
import math
import os
import random

from evaluation import Evaluator


MAX_SAMPLES = 5000


class CharacterNGramClassifier:
    def __init__(self, n=3):
        self.n = n
        self.class_ngram_counts = {}
        self.class_total_ngrams = {}
        self.class_doc_counts = {}
        self.vocabulary = set()
        self.labels = []

    def clean_text(self, text):
        return text.lower().strip()

    def get_ngrams(self, text):
        text = self.clean_text(text)

        if len(text) < self.n:
            return [text] if text else []

        ngrams = []

        for i in range(len(text) - self.n + 1):
            ngram = text[i:i + self.n]
            ngrams.append(ngram)

        return ngrams

    def fit(self, texts, labels):
        self.labels = sorted(list(set(labels)))

        for label in self.labels:
            self.class_ngram_counts[label] = {}
            self.class_total_ngrams[label] = 0
            self.class_doc_counts[label] = 0

        for text, label in zip(texts, labels):
            self.class_doc_counts[label] += 1

            ngrams = self.get_ngrams(text)

            for ngram in ngrams:
                self.vocabulary.add(ngram)

                if ngram not in self.class_ngram_counts[label]:
                    self.class_ngram_counts[label][ngram] = 0

                self.class_ngram_counts[label][ngram] += 1
                self.class_total_ngrams[label] += 1

    def predict_one(self, text):
        if not self.labels:
            raise RuntimeError("Model is not trained yet.")

        ngrams = self.get_ngrams(text)

        vocab_size = len(self.vocabulary)
        total_docs = sum(self.class_doc_counts.values())

        best_label = None
        best_score = None

        for label in self.labels:
            class_probability = self.class_doc_counts[label] / total_docs

            score = math.log(class_probability)

            for ngram in ngrams:
                count = self.class_ngram_counts[label].get(ngram, 0)

                probability = (
                    count + 1
                ) / (
                    self.class_total_ngrams[label] + vocab_size
                )

                score += math.log(probability)

            if best_score is None or score > best_score:
                best_score = score
                best_label = label

        return best_label

    def predict(self, texts):
        predictions = []

        for text in texts:
            prediction = self.predict_one(text)
            predictions.append(prediction)

        return predictions


def load_imdb_dataset(filename, max_samples=None):

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Dataset not found at '{filename}'"
        )

    texts = []
    labels = []

    with open(filename, "r", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:

            text = row["review"].strip()
            sentiment = row["sentiment"].strip().lower()

            if not text or sentiment not in ("positive", "negative"):
                continue

            label = 1 if sentiment == "positive" else 0

            texts.append(text)
            labels.append(label)

            if max_samples is not None and len(texts) >= max_samples:
                break

    if not texts:
        raise ValueError("Dataset is empty.")

    return texts, labels


def train_test_split(texts, labels, test_size=0.2, seed=42):

    combined = list(zip(texts, labels))

    random.seed(seed)
    random.shuffle(combined)

    split_index = int(len(combined) * (1 - test_size))

    train_data = combined[:split_index]
    test_data = combined[split_index:]

    train_texts, train_labels = zip(*train_data)
    test_texts, test_labels = zip(*test_data)

    return (
        list(train_texts),
        list(test_texts),
        list(train_labels),
        list(test_labels)
    )


def main():

    dataset_file = "data/IMDB Dataset.csv"

    print("Loading dataset...")

    texts, labels = load_imdb_dataset(
        dataset_file,
        max_samples=MAX_SAMPLES
    )

    print(f"Total samples loaded: {len(texts)}")

    print("\nSplitting data...")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2
    )

    print(f"Train samples: {len(train_texts)}")
    print(f"Test samples : {len(test_texts)}")

    print("\nTraining Character-Level N-Gram model (n=3)...")

    model = CharacterNGramClassifier(n=3)

    model.fit(train_texts, train_labels)

    print(f"Vocabulary size: {len(model.vocabulary)}")

    print("\nPredicting on test set...")

    predictions = model.predict(test_texts)

    evaluator = Evaluator(test_labels, predictions)

    tp, fp, fn, tn = evaluator.confusion_matrix()

    print("\n===== Results =====")

    print(f"Accuracy  : {evaluator.accuracy():.4f}")
    print(f"Precision : {evaluator.precision():.4f}")
    print(f"Recall    : {evaluator.recall():.4f}")
    print(f"F1 Score  : {evaluator.f1_score():.4f}")

    print("\nConfusion Matrix")
    print(f"TP = {tp}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print(f"TN = {tn}")


if __name__ == "__main__":
    main()