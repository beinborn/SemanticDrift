import pickle
import os
from util.evaluate_RSA import get_dists
from matplotlib import pyplot as plt

import numpy as np

# This code reproduces Figure 1 in the paper.

# Set folders
data_dir = "../data/word-based/embeddings/"
save_dir = "../results/word-based/"
os.makedirs(save_dir, exist_ok=True)

# Initialize variables
embeddings_name = "Swadesh"
languages = ["en","es", "ru"]
words ={"en": [ "small", "short", "child", "wife", "mother"],
        "es":["pequeño ", "corto ", "niño ", "esposa", "madre"],
        "ru":["небольшие", "короткий", "ребёнок", "жена", "мать"]}
vectors = []
langdicts = []


plt.figure(1)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))

i = 1
for lang in languages:
    ax = plt.subplot(1, 3, i)
    i +=1
    print("Read vectors for " + lang)
    with open(data_dir + embeddings_name + "_embeddings." + lang + ".pickle", 'rb') as handle:
        langdict = pickle.load(handle)
    langdicts.append(langdict)

    # get vectors
    langvectors = []
    for word in words["en"]:
        langvectors.append(langdict[word])
    vectors.append(langvectors)

    # Calculate cosine similarities for selected words
    x, C = get_dists([langvectors], labels=[lang], ticklabels=words[lang], distance="cosine", save_dir=save_dir)


    ### PLOTTING
    data = C[0]
    mask = np.tri(data.shape[0], k=-1)
    data = np.ma.array(data, mask=mask)
    # Plot the heatmap
    im = ax.imshow(data, cmap="YlOrRd", vmin = 0, vmax =1)

    # Finetuning the plot
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    # Label matrix with words
    ax.set_xticklabels(words[lang], fontsize =18)
    ax.set_yticklabels(words[lang],  fontsize =16)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    ax.set_xticks(np.arange(0, data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(0, data.shape[0] + 1) - 0.5, minor=True)


fig.subplots_adjust(top = 0.99, bottom=0.01,right = 0.8,  wspace=0.3)

cbar_ax = fig.add_axes([0.875, 0.32, 0.015, 0.37])
cbar_ax.tick_params(axis='y', labelsize=12)
fig.colorbar(im, cax=cbar_ax)
fig.savefig(save_dir + "selectedWords.png")

