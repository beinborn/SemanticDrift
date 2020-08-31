import numpy as np
from matplotlib import pyplot as plt

from util.evaluate_RSA import get_dists

data_dir = '../data/sentence-based/embeddings/part/'
target_langs = ["de", "nl", "fr", "es", "bg", "da", "cs", "pl", "pt", "it", "ro", "sk", "sl", "sv", 'lt', 'lv']

start = 400
end = 600

dim = 1024
target_embeddings = []
for l in target_langs:
    X = np.fromfile(data_dir + l + "_embeddings.raw", dtype=np.float32, count=-1)
    X.resize(X.shape[0] // dim, dim)
    X = X[start:end]
    target_embeddings.append(X)

# Get English sentences
with open(data_dir + "en_sents.txt") as f:
    content = f.readlines()
content = [x.strip() for x in content]
content = content[start:end]

x, C = get_dists(target_embeddings, labels=target_langs, ticklabels=content, distance="cosine")

pairs = {}
for sent1 in range(0, len(content)):
    for sent2 in range(0, len(content)):
        if sent1 < sent2:
            similarities = np.array([C[lang][sent1][sent2] for lang in range(0, len(C))])
            pairs[(content[sent1], content[sent2])] = similarities

var = {}
for key in pairs.keys():
    var[key] = np.var(pairs[key])

print("Pairs with variation across languages:")
sortedKeys = sorted(var, key=var.get, reverse=True)
plt.figure(1)
i = 1
r = 150
col = 1
indices = [0, 1, 2]
fig, axes = plt.subplots(nrows=len(indices), ncols=col)

colors = ["m", "c", "g"]
plt.tight_layout()
n = 0
for key in sortedKeys[:r * col]:
    if n in indices:
        ax = plt.subplot(len(indices), col, i)
        if len(colors) < i:
            color = "b"
        else:
            color = colors[i - 1]
        ax.plot(target_langs, pairs[key], linestyle='dashed', color=color)
        plt.title(", ".join([key[0], key[1]]), fontsize=10)
        plt.ylabel("Cos Similarity", fontsize=8)
        ax.set_ylim(0.0, 1.0)
        plt.yticks(np.arange(0, 1, step=0.2), fontsize=8)
        print(key, var[key])
        i += 1
    n += 1

plt.show()

print("\n\nPairs with lowest standard deviation across languages:")
for key in sortedKeys[-15:]:
    print(key, var[key])

print("\n\nPairs with highest standard deviation across languages:")
n = 0
for key in sortedKeys[0:150]:
    print("n= ", n, key, var[key])
    n += 1
