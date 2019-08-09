import os
import pickle
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from util.evaluate_RSA import get_dists, compute_distance_over_dists

# This code contains an additional experiment when combining the Swadesh and the Pereira List
# Quantitative results can be found in Table 2, and the Figure of the tree in the Appendix.
data_dir = "data/word-based/embeddings/"
save_dir = "results/word-based/"
os.makedirs(save_dir, exist_ok=True)

# All languages
languages = ["it", "pt", "es", "ca", "fr", "en", "nl", "de", "sv", "da", "no", "fi", "et", "ru", "uk", "mk", "bg", "ro",
             "pl", "cs", "sk", "sl", "hr", "hu", "el", "he", "tr", "id"]


# Languages for the gold tree
# languages = ["it", "pt", "es", "ca", "fr", "en", "nl", "de", "sv", "da", "no", "ru", "uk", "bg", "ro", "pl", "cs", "sk",
#              "sl", "hr"]
# Make sure to adjust the names for saving the files , so that you do not get confused with the results for all languages.
print(len(languages))

words = set()
vectors = []

for lang in languages:
    name = "swadesh"
    with open(data_dir + name + "_embeddings." + lang + ".pickle", 'rb') as handle:
        swadesh_dict = pickle.load(handle)
        if len(words) == 0:
            words.update(swadesh_dict.keys())

    name = "pereira"
    with open(data_dir + name + "_embeddings." + lang + ".pickle", 'rb') as handle:
        pereira_dict = pickle.load(handle)
        if len(words) < 300:
            words.update(pereira_dict.keys())

    combined_dict = {**swadesh_dict, **pereira_dict}
    print(combined_dict.keys())
    print(len(words))
    print(len(words) == len(combined_dict.keys()))

    # get vectors
    langvectors = []
    for word in words:
        langvectors.append(combined_dict[word])
    vectors.append(langvectors)

### Representational Similarity Analysis
# Calculate similarity matrices for all languages
x, C = get_dists(vectors, labels=languages, ticklabels=words, distance="cosine", save_dir=save_dir + name)

# Calculate RSA over all languages
distance_matrix = compute_distance_over_dists(x, C, languages, save_dir=save_dir + name)

#
# Save distance matrix
with open(save_dir + "distancematrix_combinedstimuli.pickle", 'wb') as handle:
    pickle.dump(distance_matrix, handle)

# ### CLUSTERING
method = "ward"

cluster_results = linkage(pdist(distance_matrix), method=method)

print(cluster_results)

### PLOT CLUSTER RESULT
name = "Combined Stimuli"
plt.figure(figsize=(25, 5))
plt.title(name, fontsize=24)

plt.ylabel('Distance', fontsize=20)
plt.tick_params(labelsize=16)

# Make sure that the color coding for clusters is comparable across plots
threshold = 0.72
hierarchy.set_link_color_palette(['c','g', 'y', 'r', 'm', 'c', 'b'])


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
