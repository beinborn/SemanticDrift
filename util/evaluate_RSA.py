import numpy as np

import scipy as sp
import scipy.spatial
import scipy.stats
import logging

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_dists(data, labels=None, ticklabels=None, distance="euclidean", save_dir="plots/"):
    if labels is None:
        labels = []
    logging.info("Calculating dissimilarity matrix")
    x = {}
    C = {}

    # For each list of vectors
    for i in np.arange(len(data)):
        x[i] = data[i]

        # Calculate distances between vectors
        #print("Calculating cosine for: " + labels[i])
        C[i] = 1 - (sp.spatial.distance.cdist(x[i], x[i], distance) + 0.00000000001)
        #print("Normalizing")
        # Normalize
        C[i] /= C[i].max()

    # Uncomment this if you want to plot the matrices for all languages. This might be useful for more detailed analyses.
    # for i in C:
    #     print(C[i].shape)
    #     print("Start plotting")
        # if len(ticklabels) == 0:
        #     ticklabels = [x for x in range(1, len(C[i]) + 1)]
        # print(ticklabels)
        # print(save_dir)
        # print(C[i].shape)
        # print(C[i][ 0:40,0:40].shape)
        # # Only plot the first 40 words
        # fig = get_plot(C[i][0:40,0:40], ticklabels[0:40], labels[i], cbarlabel=(distance.capitalize() + " Similarity"))
        # fig.savefig(save_dir + "RDM_" + labels[i] + ".png")


    return x, C


# Compare two or more RDMs
def compute_distance_over_dists(x, C, labels, save_dir="plots/RSA_"):
    logging.info("Calculate correlation over RDMs")
    keys = np.asarray(list(x.keys()))

    # We calculate three different measures.
    spearman = np.zeros((len(keys), len(keys)))
    # pearson = np.zeros((len(keys), len(keys)))
    for i in np.arange(len(keys)):
        for j in np.arange(len(keys)):
            corr_s = []
            corr_p = []
            for a, b in zip(C[keys[i]], C[keys[j]]):
                s, _ = sp.stats.spearmanr(a, b)
                p, _ = sp.stats.pearsonr(a, b)
                corr_s.append(s)
                corr_p.append(p)
            spearman[i][j] = np.mean(corr_s)
            # If you prefer Pearson correlation
            # pearson[i][j] = np.mean(corr_p)

    #print(spearman)
    # Uncomment this, if you want to plot the matrix and save it.
    # name = save_dir.split("/")[-1]
    im, cbar = get_plot(spearman, labels, cbarlabel="Spearman Correlation")
    plt.savefig(save_dir + "RSA_Spearman.png")
    # Save the matrix
    write_matrix(spearman, save_dir + "RSA_Spearman.txt", labels=labels)
    return spearman


# Code for plotting, based on code by Samira Abnar, but slightly modified

def get_plot(data, labels, title="", cbarlabel="Cosine Distance"):
    plt.rcParams["axes.grid"] = False
    plt.interactive(False)
    print(labels)
    fig, ax = plt.subplots(figsize=(20, 20))

    im, cbar= heatmap(data, labels, labels, ax=ax,
                       cmap="YlOrRd", vmin = 0.2, vmax =1, title=title, cbarlabel=cbarlabel)

    return fig, cbar


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", title="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Only plot upper half
    mask = np.tri(data.shape[0], k=-1)
    data = np.ma.array(data, mask=mask)
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    ax.set_title(title, pad=50.0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    # create an axis on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.1 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=3.6)
    cbar = ax.figure.colorbar(im, ax=ax, cax=cax, **cbar_kw)
    cax.set_ylabel(cbarlabel, fontsize=40, labelpad=40)
    cax.tick_params(axis='y', labelsize=36)
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=48)
    ax.set_yticklabels(row_labels, fontsize=48)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    ax.set_xticks(np.arange(0, data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(0, data.shape[0] + 1) - 0.5, minor=True)

    return im, cbar


# Save the matrix to a text file for inspecting it.
# Based on a code snippet provided here: https://stackoverflow.com/questions/3685265/how-to-write-a-multidimensional-array-to-a-text-file
def write_matrix(data, filename, labels=[]):
    with open(filename, "w") as f:
        f.write('# Array shape: {0}\n'.format(data.shape))
        if len(labels) > 0:
            for label in labels:
                f.write(str(label) + "\t")
            f.write("\n")
        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        i = 0
        print(len(data), len(labels))
        for data_slice in data:
            if len(labels) > 0:
                f.write(str(labels[i]) + "\t")
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(f, data_slice, fmt='%-5.3f', newline="\t")

            # Writing out a break to indicate different slices...
            f.write('\n')
            i += 1
