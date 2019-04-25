from scipy.cluster.hierarchy import _plot_dendrogram
from matplotlib import pyplot as plt

import numpy as np

# These lines hard-code the gold tree by Rabinovich et al.

save_dir = "results/sentence-based/"

# Coordinates for the tree
icoord = [[5.0, 5.0, 15.0, 15.0], [25.0, 25.0, 35.0, 35.0], [30.0, 30.0, 45.0, 45.0], [10.0, 10.0, 37.5, 37.5],
          [55.0, 55.0, 65.0, 65.0], [21.25, 21.25, 60.00, 60.00], [75.0, 75.0, 85.0, 85.0], [80.0, 80.0, 95.0, 95.0],
          [87.5, 87.5, 105.0, 105.0], [95.0, 95.0, 115.0, 115.0], [125.0, 125.0, 135.0, 135.0],
          [145.0, 145.0, 155.0, 155.0], [130.0, 130.0, 150.0, 150.0], [140.0, 140.0, 165.0, 165.0],
          [100, 100, 152.5, 152.5], [37.5, 37.5, 126.6, 126.5]]

dcoord = [[0.0, 0.95, 0.95, 0.0], [0.0, 0.4, 0.4, 0.0], [0.4, 0.7, 0.7, 0.0], [0.95, 1.0, 1.0, 0.7],
          [0.0, 0.7, 0.7, 0.0], [1.0, 2.0, 2.0, 0.7], [0.0, 0.5, 0.5, 0.0], [0.5, 0.65, 0.65, 0.0],
          [0.65, 0.9, 0.9, 0.0], [0.9, 1.3, 1.3, 0.0], [0.0, 0.675, 0.675, 0.0], [0.0, 0.55, 0.55, 0.0],
          [0.675, 1.2, 1.2, 0.55], [1.2, 1.25, 1.25, 0.0], [1.3, 2.7, 2.7, 1.25], [2.0, 3.0, 3.0, 2.7]]

# Set leaves
ivl = ['bg', 'sl', 'cs', 'sk', 'pl', "lt", "lv", "pt", "es", "it", "fr", "ro", "nl", "de", "da", "sv", "en"]

# Prepare Plot
threshold = 1.7
fig = plt.figure(figsize=(25, 10))
plt.title("Gold Tree", fontsize=20)
plt.yticks([])
plt.tick_params(labelsize=16)
n = 18
mh = 3
color_list = ['c', 'c', 'c', 'c', 'g', 'k', 'r', 'r', 'r', 'r', 'm', 'm', 'm', 'm', 'k', 'k']
p = 30
orientation = 'top'
no_labels = False
print("Plotting")
plt.tight_layout()
_plot_dendrogram(icoord, dcoord, ivl, p, n, mh, orientation,
                 no_labels, color_list,
                 leaf_font_size=22.,
                 leaf_rotation=None,
                 contraction_marks=None,
                 ax=None,
                 above_threshold_color="k")

plt.savefig(save_dir + "goldtree" + str(int(threshold * 100)) + ".png")

# Rabinovych tree
# Coordinates for the tree

icoord = [[5.0, 5.0, 15.0, 15.0], [10.0, 10.0, 25.0, 25.0], [17.5, 17.5, 35.0, 35.0], [45.0, 45.0, 55.0, 55.0],
          [50.0, 50.0, 65.0, 65.0], [26.25, 26.25, 57.5, 57.5], [75.0, 75.0, 85.0, 85.0],
          [41.87, 41.87, 80.0, 80.0], [95.0, 95.0, 105.0, 105.0], [100.0, 100.0, 115.0, 115.0],
          [125.0, 125.0, 135.0, 135.0], [107.5, 107.5, 130.0, 130.0], [145.0, 145.0, 155.0, 155.0],
          [150.0, 150.0, 165.0, 165.0], [118.75, 118.75, 157.5, 157.5], [60.93, 60.93, 138.13, 138.13]]

dcoord = [[0.0, 0.5, 0.5, 0.0], [0.5, 1.0, 1.0, 0.0], [1.0, 1.5, 1.5, 0.0], [0.0, 0.5, 0.5, 0], [0.5, 1.0, 1.0, 0.0],
          [1.5, 2.0, 2.0, 1.0], [0.0, 0.5, 0.5, 0.0], [2.0, 2.5, 2.5, 0.5], [0.0, 0.5, 0.5, 0.0], [0.5, 1.0, 1.0, 0.0],
          [0.0, 0.5, 0.5, 0.0], [1.0, 1.5, 1.5, 0.5], [0.0, 0.5, 0.5, 0.0], [0.5, 1.0, 1.0, 0.0], [1.5, 2.0, 2.0, 1.0],
          [2.5, 3.0, 3.0, 2.0]]

# Set leaves
ivl = ["sl", "pl", "lt", 'bg', 'sl', 'cs', "pt", "lt", "ro", "da", "sv", "en", "nl", "de", "es", "fr", "it"]
# Prepare Plot
threshold = 1.5

plt.title("Rabinovich Tree", fontsize=20)
plt.yticks([])
plt.tick_params(labelsize=16)
n = 18
mh = 3
color_list = ['c', 'c', 'c', 'c', 'c','c', 'g', 'k', 'r', 'r', 'r', 'r', 'm', 'm', 'k', 'k']
p = 30
orientation = 'top'
no_labels = False
print("Plotting")
plt.tight_layout()
_plot_dendrogram(icoord, dcoord, ivl, p, n, mh, orientation,
                 no_labels, color_list,
                 leaf_font_size=22.,
                 leaf_rotation=None,
                 contraction_marks=None,
                 ax=None,
                 above_threshold_color="k")

plt.savefig(save_dir + "rabinovichtree" + str(int(threshold * 100)) + ".png")
