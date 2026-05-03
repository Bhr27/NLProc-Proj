dataset = [
    ("free money now", "spam"),
    ("win money now", "spam"),
    ("call me now", "normal"),
    ("let's meet now", "normal")
]

def tokenize(text):
    return text.lower().split()

def build_vocab(dataset):
    vocab = set()
    for text, _ in dataset:
        tokens = tokenize(text)
        vocab.update(tokens)
    return vocab


vocab = build_vocab(dataset)

total_words = {"spam": 0, "normal": 0}
class_counts = {"spam": 0, "normal": 0}
word_counts = {"spam": {}, "normal": {}}

for text, label in dataset:
    class_counts[label] += 1
    tokens = tokenize(text)

    for token in tokens:
        total_words[label] += 1
        if token not in word_counts[label]:
            word_counts[label][token] = 0
        word_counts[label][token] += 1

        print("token:", token)
        print("word_counts:", word_counts)

    print("total_words:", total_words)
    print("-")

    priors = {}
total_docs = len(dataset)

for label in class_counts:
    priors[label] = class_counts[label] / total_docs

print("priors:", priors)

likelihoods = {"spam": {}, "normal": {}}
V = len(vocab)  

for label in ["spam", "normal"]:
    for word in vocab:
        word_count = word_counts[label].get(word, 0)
        likelihoods[label][word] = (word_count + 1) / (total_words[label] + V)

        print(" likelihood:", likelihoods[label][word])
        print("likelihoods so far:", likelihoods[label])
        print("-")

    def score(text, label):
     tokens = tokenize(text)
    prob = priors[label]

def score(text, label):
    tokens = tokenize(text)
    prob = priors[label]

    for token in tokens:
        if token in vocab:
            prob *= likelihoods[label][token]
        else:
            prob *= 1 / (total_words[label] + V)

    return prob

print(score("free money", "spam"))
print(score("free money", "normal"))

def predict(text):
    scores = {}
    for label in ["spam", "normal"]:
        scores[label] = score(text, label)

    return max(scores, key=scores.get)

print("prediction:", predict("free money"))
