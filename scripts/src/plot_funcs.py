# plot_funcs.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Plots images in 2 rows: top row is query, bottom row is answer
def plot_query_retrieval(img_query, result_images, outfile, N):
    """
    Plot the input image along with the 5 recommended images
    Input:
        img_query:      user-input image
        result_images:  tuple: (imgs retrieved, artist handles, studio names)
        outfile:        path and name of saved image
        N:              number of retrieved images to plot
    """
    imgs_retrieval = result_images[0]
    artist_handles = result_images[1]
    studio_names = result_images[2]

    n_retrieval = N #len(imgs_retrieval)
    fig = plt.figure(figsize=(2*n_retrieval, 4))
    fig.suptitle("Similar images (k={})".format(n_retrieval), fontsize=25)

    # Plot query image
    ax = plt.subplot(2, n_retrieval, 0 + 1)
    plt.imshow(img_query)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(4)  # increase border thickness
        ax.spines[axis].set_color('black')  # set to black
    ax.set_title("Input",  fontsize=14)  # set subplot title

    # Plot retrieval images
    for i, img in enumerate(imgs_retrieval):
        ax = plt.subplot(2, n_retrieval, n_retrieval + i + 1)
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1)  # set border thickness
            ax.spines[axis].set_color('black')  # set to black
#         ax.set_title("Rank #%d" % (i+1), fontsize=14)  # set subplot title
        ax.set_title(studio_names[i], fontsize=10)  # set subplot title

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, bbox_inches='tight')
    plt.close()
