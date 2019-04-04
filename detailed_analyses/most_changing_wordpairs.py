import pickle
from util.evaluate_RSA import get_dists
from matplotlib import pyplot as plt
import numpy as np
import os

# This code reproduces Figure 3 of the paper

# Set directories
data_dir = "data/word-based/embeddings/"
save_dir = "results/word-based/"
os.makedirs(save_dir, exist_ok=True)

# Initialize variables
languages = ["it", "pt", "es", "ca", "fr", "en", "nl", "de", "sv", "da", "no", "fi", "et", "ru", "uk", "mk", "bg", "ro",
             "pl", "cs", "sk", "sl", "hr", "hu", "el", "he", "tr", "id"]
langdicts = []
words = []
vectors = []

embeddings_name = ["Pereira", "Swadesh"]


for lang in languages:
    print("Read vectors for " + lang)

    with open(data_dir + embeddings_name[0] + "_embeddings." + lang + ".pickle", 'rb') as handle:
        langdict1 = pickle.load(handle)
    with open(data_dir + embeddings_name[1] + "_embeddings." + lang + ".pickle", 'rb') as handle:
        langdict2 = pickle.load(handle)

    # For this analysis, we take the union of both lists
    if len(words) == 0:
        words = list(set(list(langdict1.keys())).union(list(langdict2.keys())))
        print(words)

    # get vectors
    langvectors = []
    for word in words:
        try:
            langvectors.append(langdict1[word])
        except KeyError:
            langvectors.append(langdict2[word])
            pass
    vectors.append(langvectors)

# Calculate cosine similarities
x, C = get_dists(vectors, labels=languages, ticklabels=words, distance="cosine")

pairs = {}

# Extract the similarity value for a word pair for all languages
for word1 in range(0, len(words)):
    for word2 in range(0, len(words)):
        if word1 < word2:
            similarities = np.array([C[lang][word1][word2] for lang in range(0, len(C))])
            pairs[(words[word1], words[word2])] = similarities

# Calculate variance of similarities
var = {}
for key in pairs.keys():
    var[key] = np.var(pairs[key])

print("Pairs with variation across languages:")
sortedKeys = sorted(var, key=var.get, reverse=True)
plt.figure(1)
i = 1
r = 3
col = 1
fig, axes = plt.subplots(nrows=r, ncols=col)

colors = ["m", "c", "g"]
plt.tight_layout()
for key in sortedKeys[:r * col]:
    ax = plt.subplot(r, col, i)
    if len(colors) < i:
        color = "b"
    else:
        color = colors[i - 1]
    ax.plot(languages, pairs[key], linestyle='dashed', color=color)
    plt.title(", ".join([key[0], key[1]]), fontsize=10)
    plt.ylabel("Cos Similarity", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    plt.yticks(np.arange(0, 1, step=0.2), fontsize=8)
    print(key, var[key])
    i += 1

plt.show()
plt.savefig(save_dir +"mostVaryingPairs.png")


# Just out of curiosity
print("\n\nPairs with lowest variance across languages:")
for key in sortedKeys[-15:]:
    print(key, var[key])
