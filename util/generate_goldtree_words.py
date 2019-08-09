from scipy.cluster.hierarchy import _plot_dendrogram
from matplotlib import pyplot as plt

import numpy as np

# These lines hard-code the gold tree for words


save_dir = "../results/word-based/"

# Coordinates for the tree
#Position
icoord = [

    # bg-sl
    [5.0, 5.0, 15.0, 15.0],

    # ru-uk
    [25.0, 25.0, 35.0, 35.0],
    # cs/sk
    [45.0, 45.0, 55.0, 55.0],
    # cs/sk -hr
    [50.0, 50.0, 65.0, 65.0],
    # ru/uk -  cs/sk/hr
    [30.0, 30.0, 57.5, 57.5],
    # ru-tree -pl
    [43.75, 43.75, 75.0, 75.0],
    # bg/sl - pl
    [10.00, 10.00, 59.375, 59.375],
    # pt-es
    [85.0, 85.0, 95.0, 95.0],
    # pt/es - it
    [90.0, 90.0, 105.0, 105.0],
    # it - ca
    [97.5, 97.5, 115.0, 115.0],
    # ca - fr
    [106.25, 106.25, 125.0, 125.0],
    # fr-ro
    [115.625, 115.625, 135.0, 135.0],
    # nl-de
    [145.0, 145.0, 155.0, 155.0],
    # da-no
    [165.0, 165.0, 175.0, 175.0],
    # da/no-sv
    [170.0, 170.0, 185.0, 185.0],
    # nl/de - scandinavian
    [150.0, 150.0, 177.5, 177.5],
    # germanic-en
    [163.75, 163.75, 195.0, 195.0],
    # romanic-germanic/en
    [125.31, 125.31, 179.375, 179.375],
    # eastern - western (top edge)
    [34.69, 34.69, 152.34, 152.34]
]

# Height
dcoord = [
    # bg-sl
    [0.0, 0.95, 0.95, 0.0],
    # ru-uk
    [0.0, 0.6, 0.6, 0.0],
    # cs/sk
    [0.0, 0.4, 0.4, 0.0],
    # cs/sk -hr
    [0.4, 0.65, 0.65, 0.0],
    # ru/uk -  cs/sk/hr
    [0.6, 0.7, 0.7, 0.65],
    # ru-tree -pl
    [0.7, 0.75, 0.75, 0.0],
    # bg/sl - pl
    [0.95, 1.0, 1.0, 0.75],
    # pt-es
    [0.0, 0.5, 0.5, 0.0],
    # pt-es - it
    [0.5, 0.65, 0.65, 0.0],
    # it - ca
    [0.65, 0.75, 0.75, 0.0],
    # ca - fr
    [0.75, 0.9, 0.9, 0.0],
    # fr-ro
    [0.9, 1.25, 1.25, 0.0],
    # nl-de
    [0.0, 0.675, 0.675, 0.0],
    # da-no
    [0.0, 0.3, 0.3, 0.0],
    # da/no-sv
    [0.3, 0.55, 0.55, 0.0],
    # nl/de - scandinavian
    [0.675, 1.2, 1.2, 0.55],
    # germanic-en
    [1.2,1.25, 1.25, 0.0],
    # romanic-germanic/en
    [1.25, 1.5, 1.5, 1.25],
     # eastern - western
    [1.0, 1.7, 1.7, 1.5]

]

# Set leaves
ivl = ["bg", "sl", "ru", "uk", "cs", "sk","hr","pl", "pt", "es", "it", "ca","fr", "ro", "nl", "de", "da","no", "sv", "en"]

# Prepare Plot
threshold = 1.3
fig = plt.figure(figsize=(25, 5))
plt.title("Gold Tree", fontsize=26)
plt.yticks([])
plt.tick_params(labelsize=16)

n = 18
mh = 2
color_list = ['c', 'c', 'c', 'c', 'c', 'c', 'c' , 'r', 'r', 'r', 'r','r' ,'m', 'm', 'm', 'm','m', 'k', 'k','k','k','k']
p = 30
orientation = 'top'
no_labels = False
print("Plotting")
plt.tight_layout()
plt.ylabel('', fontsize=20)

with plt.rc_context({'lines.linewidth': 2.0}):
    _plot_dendrogram(icoord, dcoord, ivl, p, n, mh, orientation,
                     no_labels, color_list,
                     leaf_font_size=26.,
                     leaf_rotation=None,
                     contraction_marks=None,
                     ax=None,
                     above_threshold_color="k")

plt.savefig(save_dir + "goldtree_words" + str(int(threshold * 100)) + ".png")
