import os
import pickle

import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from util.evaluate_RSA import get_dists, compute_distance_over_dists

data_dir = "data/word-based/embeddings/"
save_dir = "results/word-based/"
os.makedirs(save_dir, exist_ok=True)

# Set the embeddings name to either "Swadesh" or "Pereira"
lists = ["Swadesh", "Pereira"]

# All languages
# languages = ["it", "pt", "es", "ca", "fr", "en", "nl", "de", "sv", "da", "no", "fi", "et", "ru", "uk", "mk", "bg", "ro",
#             "pl", "cs", "sk", "sl", "hr", "hu", "el", "he", "tr", "id"]


# Languages for the gold tree
languages =  ["it", "pt", "es", "ca", "fr", "en", "nl", "de", "sv", "da", "no",  "ru", "uk",  "bg", "ro", "pl", "cs", "sk", "sl", "hr"]
print(len(languages))
# If you want to reproduce figure 2:   RSA for pt, es, fr, de, fi
#languages = ["pt","es","fr", "de", "fi"]
for name in lists:
    print(name)

    langdicts = []
    words = []
    vectors = []

    for lang in languages:
        print("Read vectors for " + lang)
        with open(data_dir + name + "_embeddings." + lang + ".pickle", 'rb') as handle:
            langdict = pickle.load(handle)
        langdicts.append(langdict)
        print(langdict.keys())
        # get words
        if len(words) == 0:
            words = list(langdicts[0].keys())

        # get vectors
        langvectors = []
        for word in words:
            langvectors.append(langdict[word])
        vectors.append(langvectors)

    ### Representational Similarity Analysis
    # Calculate similarity matrices for all languages
    x, C = get_dists(vectors, labels=languages, ticklabels=words, distance="cosine", save_dir=save_dir+name)

    # Calculate RSA over all languages
    distance_matrix = compute_distance_over_dists(x, C, languages, save_dir=save_dir+name)


    #
    # Save distance matrix
    with open(save_dir + "ComparisonToGold_distancematrix_" + name +".pickle", 'wb') as handle:
        pickle.dump(distance_matrix, handle)
    #
    # ### CLUSTERING
    method = "ward"

    cluster_results = linkage(pdist(distance_matrix), method=method)

    print(cluster_results)

    ### PLOT CLUSTER RESULT

    plt.figure(figsize=(25, 5))
    plt.title(name, fontsize=24)

    plt.ylabel('Distance', fontsize=20)
    plt.tick_params(labelsize=16)

    # Make sure that the color coding for clusters is comparable across plots
    if name == "Swadesh":
        threshold = 0.73
        hierarchy.set_link_color_palette(['c', 'y', 'g', 'r', 'm', 'b'])
    else:
        hierarchy.set_link_color_palette(['g', 'y', 'r', 'm', 'c', 'b'])
        threshold = 0.7

    #When comparing to gold tree, adjust the color coding by uncommenting this:
    threshold = 0.9
    hierarchy.set_link_color_palette(['c', 'r', 'm', 'b'])
    plt.yticks([])
    plt.ylabel('', fontsize=20)
    # Increase edge thickness
    with plt.rc_context({'lines.linewidth': 2.0}):
        results = dendrogram(
            cluster_results,
            labels=languages,
            # link_color_func=lambda k: getColor(k),
            count_sort=False,
            # sistance_sort = True,
            leaf_font_size=20.,
            color_threshold=threshold,
            above_threshold_color='k'

        )

    print(cluster_results)
    plt.savefig(save_dir + "ComparisonToGold_Dendrogram_" + name + "_" + method + str(int(threshold * 100)) + ".png")

# TODO: get results for combining both lists