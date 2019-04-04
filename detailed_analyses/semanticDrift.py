import os
from util.evaluate_RSA import get_dists
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pickle
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from util.muse_utils import load_vec

# This code is used for the identification of semantic drift.
# It reproduces Figure 5 of the paper.


# Set directories
data_dir = "../data/word-based/embeddings/"
save_dir = "../results/word-based/"
os.makedirs(save_dir, exist_ok=True)
embedding_dir = '/Users/lisa/Corpora/embeddings/multilingual/'

nmax = 200000  # maximum number of word embeddings to load
# Selected languages
# Note: when we created the plot, we erroneously included English in the set of languages and only noticed it later.
# This changes the similarity vector and has an effect on the results.
# We leave it in here for reproducibility, but it is not correct.
languages = ["en", "es", "pt", "de", "nl", "ru", "uk"]

langdicts = []
words = []
all_vectors = []

embeddings_name = ["Pereira"]
for lang in languages:
    print("Read vectors for " + lang)

    with open(data_dir + embeddings_name[0] + "_embeddings." + lang + ".pickle", 'rb') as handle:
        langdict = pickle.load(handle)

    if len(words) == 0:
        words = list(langdict.keys())

    # get vectors
    langvectors = []
    for word in words:
        langvectors.append(langdict[word])

    all_vectors.append(langvectors)

x, C = get_dists(all_vectors, labels=languages, ticklabels=words, distance="cosine")

meanCorrelations = {}
clusters = [[1, 2], [3, 4], [5, 6]]
for word in range(0, len(words)):
    within_cluster_correlations = []
    outside_cluster_correlations = []

    for lang1 in range(1, len(languages)):
        for lang2 in range(1, len(languages)):
            if lang1 < lang2:
                # Determine if the languages are in the same cluster
                sameCluster = False
                for cluster in clusters:
                    if lang1 in cluster and lang2 in cluster:
                        sameCluster = True

                # Calculate correlations
                if sameCluster:
                    within_cluster_correlations.append(spearmanr(C[lang1][word], C[lang2][word])[0])
                else:
                    outside_cluster_correlations.append(spearmanr(C[lang1][word], C[lang2][word])[0])

    # Calculate means
    mean_within = np.mean(np.asarray(within_cluster_correlations))
    mean_outside = np.mean(np.asarray(outside_cluster_correlations))
    meanCorrelations[word] = (mean_within - mean_outside)

sortedKeys = sorted(meanCorrelations, key=meanCorrelations.get, reverse=True)[:15]
print(sortedKeys)
print([words[key] for key in sortedKeys])
print([meanCorrelations[key] for key in sortedKeys])

### PLOTTING ###

# Read in full embeddings for applying PCA
# embeddings = []
# for lang in languages[0:]:
#     langpath = embedding_dir + "wiki.multi." + lang + ".vec"
#     lang_embeddings, _, _ = load_vec(langpath, nmax)
#     embeddings.append(lang_embeddings)
#
# pca = PCA(n_components=2)
# print("Fit PCA")
# flat_embeddings = []
# for emb in embeddings:
#     flat_embeddings.extend(emb)
#
# pca.fit(flat_embeddings)
k = 4
# with open(save_dir + "pca.pickle", 'wb') as handle:
#     pickle.dump(pca, handle)

with open(save_dir + "pca.pickle", 'rb') as handle:
    pca = pickle.load(handle)
for word in sortedKeys:
    if words[word] == "lady" or words[word] == "reaction":
        vectors = []
        labels = []
        print(word)
        for langid in range(1, len(languages)):
            similarities = C[langid][word]
            # Get the k neighbours with the highest similarities
            sortedNeighbours = np.argsort(similarities)
            print("High Similarities: ")
            print([similarities[n] for n in sortedNeighbours[-k:]])
            print("Low Similarities: ")
            print([similarities[n] for n in sortedNeighbours[:k]])

            neighbours = sortedNeighbours[-k:]
            print(len(neighbours), [words[w] for w in neighbours])
            labels.extend([words[w] for w in neighbours if not w == word])
            vectors.extend([all_vectors[langid][w] for w in neighbours if not w == word])

        # Get reduced vector for the neighbours
        print("Vectors")
        print(np.asarray(vectors).shape)
        print("Labels")
        print(np.asarray(labels).shape)
        vectors = pca.transform(vectors)

        print("plotting")
        # Visualize
        print(vectors)

        x = []
        y = []
        for value in vectors:
            x.append(value[0])
            y.append(value[1])
        plt.interactive(False)
        colors = [[c] * (k - 1) for c in ["r", "m", "g", "y", "b", "c"]]
        weights = ["bold" if i % k == 0 else "light" for i in range(0, k * len(languages) + 1)]
        boxstyle = [dict(boxstyle="round", facecolor="none") if i % k == 0 else None for i in
                    range(0, k * len(languages) + 1)]

        flat_colors = []
        for c in colors:
            flat_colors.extend(c)
        colors = flat_colors
        print(colors)
        print(labels)
        fig, ax = plt.subplots()
        for i in range(0,len(x)):
            ax.scatter(x[i], y[i], [80], colors[i])
            ax.annotate(labels[i],
                        xy=(x[i], y[i]),
                        xytext=(5, 2),
                        textcoords='offset points',
                        fontsize=11,
                        ha='right',
                        va='bottom',
                        # weight=weights[i],  bbox=boxstyle[i],
                        color=colors[i])

        patches = []
        for i in range(1, len(languages)):
            print(i,k)
            color = colors[(i * (k-1))-1]
            label = languages[i]
            patches.append(mpatches.Patch(color=color, label=label))

        # Define and place the legend (legend placement has to be adjusted manually for selected examples).
        legend = ax.legend(handles=patches, loc='lower left')
        plt.title("Neighbors for " + words[word])
        plt.savefig(save_dir + words[word] + "_NNs_lowerleft.png")
