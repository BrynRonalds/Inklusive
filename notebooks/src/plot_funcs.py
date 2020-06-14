# plot_funcs.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Plots images in rows: top row is query, bottom rows is answer
def plot_query_retrieval(result_images, outfile):
    """
    Plot the recommended images
    Input:
        result_images:  tuple: (imgs retrieved, artist handles, studio names)
        outfile:        path and name of saved image
    """
    imgs_retrieval = result_images[0]
    artist_handles = result_images[1]
    studio_names = result_images[2]
    
    n_retrieval = len(imgs_retrieval)
    row = 2
    col = int(np.ceil(n_retrieval/2))
    
    fig = plt.figure(dpi=256) #figsize=(2*n_retrieval, 4))
    # Plot retrieval images
    for i, img in enumerate(imgs_retrieval):
        ax = plt.subplot(row, col, n_retrieval - i)
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1)  # set border thickness
            ax.spines[axis].set_color('black')  # set to black
#         ax.set_title("Rank #%d" % (i+1), fontsize=14)  # set subplot title
        ax.set_title(studio_names[i], fontsize=8)  # set subplot title

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, bbox_inches='tight')
    plt.close()
