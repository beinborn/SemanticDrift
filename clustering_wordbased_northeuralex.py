import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from util.evaluate_RSA import get_dists, compute_distance_over_dists

data_dir = "data/word-based/embeddings/"
save_dir = "results/word-based/"
os.makedirs(save_dir, exist_ok=True)

lists = ["northeuralex"]


# Languages for the gold tree
languages =  ["it", "pt", "es", "ca", "fr", "en", "nl", "de", "sv", "da", "no",  "ru", "uk",  "bg", "ro", "pl", "cs", "sk", "sl", "hr"]
print(len(languages))

for name in lists:
    print(name)

    langdicts = []
    words = []
    vectors = []

    distance_matrix_name = save_dir + "distancematrix_" + name+ ".pickle"
    if not os.path.isfile(distance_matrix_name):
        for lang in languages:
            print("Read vectors for " + lang)
            with open(data_dir + name + "_embeddings." + lang + ".pickle", 'rb') as handle:
                langdict = pickle.load(handle)
            langdicts.append(langdict)
            print(list(langdict.keys()))
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
        with open(save_dir + "_distancematrix_" + name +".pickle", 'wb') as handle:
            pickle.dump(distance_matrix, handle)

    else:
        # If distance matrix has been calculated and you just want to adjust the plot, you can also load it directly
        with open(distance_matrix_name, 'rb') as handle:
            distance_matrix = pickle.load(handle)
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
    # threshold = 0.9
    # hierarchy.set_link_color_palette(['c', 'r', 'm', 'b'])

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
    plt.savefig(save_dir + "Dendrogram_" + name + "_" + method + str(int(threshold * 100)) + ".png")

    # # Additional analysis: check variance of similarity scores
    # upper_part = np.triu(distance_matrix,1)
    # # Flatten and remove all zeros
    # flattened = np.matrix.flatten(upper_part)
    # scores = [x for x in flattened if not x==0]
    # var = np.var(np.asarray(scores))
    # mean = np.mean(np.asarray(scores))
    # print(scores)
    # print("Variance of similarity scores: ")
    # print(name, var, mean)