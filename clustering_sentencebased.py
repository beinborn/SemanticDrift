import pickle
import os
from util.evaluate_RSA import get_dists, compute_distance_over_dists
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
import numpy as np

languages = ["en", "de", "nl", "fr", "es", "bg", "da", "cs", "pl", "pt", "it", "ro", "sk", "sl", "sv", 'lt', 'lv']
result_dir = "results/sentence-based/"
embed_data_dir = "data/sentence-based/embeddings/part/"

# Short sentences start in line 0, mid sentence in line 200, long sentence in line 400

startpoints ={"short": 1, "mid":200, "long": 400}
for category in startpoints.keys():
    start = startpoints[category]-1
    end = start +200

    distance_matrix_name = result_dir + "distancematrix_" +category +".pickle"
    if not os.path.isfile(distance_matrix_name):
        dim = 1024
        target_embeddings = []
        # Get embeddings

        for l in languages:
                X = np.fromfile(embed_data_dir+l+"_embeddings.raw", dtype=np.float32, count=-1)
                print(X.shape)
                X.resize(X.shape[0] // dim, dim)
                print(X.shape)
                X = X[start:end]
                print(X.shape)
                target_embeddings.append(X)

        # Calculate representational similarity analysis
        x, C = get_dists(target_embeddings, labels=languages, ticklabels=[], distance="cosine", save_dir=result_dir)
        distance_matrix = compute_distance_over_dists(x, C, languages, save_dir=result_dir +category)
        print(distance_matrix.shape)

        # Save result
        with open(distance_matrix_name, 'wb') as handle:
            pickle.dump(distance_matrix, handle)

    else:
        # If distance matrix has been calculated and you just want to adjust the plot, you can also load it directly
        with open(distance_matrix_name, 'rb') as handle:
            distance_matrix = pickle.load(handle)

    method = "ward"
    threshold = 0.22
    cluster_results = linkage(pdist(distance_matrix), method=method)

    plt.figure(figsize=(25, 10))
    plt.title("Europarl", fontsize=20)

    plt.ylabel('Distance', fontsize=16)
    plt.tick_params(labelsize=16)


    hierarchy.set_link_color_palette([ 'r', 'm', 'c', 'g'])


    results = dendrogram(
        cluster_results,
        labels=languages,
        count_sort=False,
        leaf_font_size=18.,
        color_threshold=threshold,
        above_threshold_color='k'

    )

    plt.savefig(result_dir + "Dendrogram_" + category+"_"+  method + str(int(threshold * 100)) + ".png")

