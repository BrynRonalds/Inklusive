# plot_funcs.py
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_image_retrieval(imgs_retrieval, outpath):
    """
    Plot the recommended images
    """
    # Plot retrieval images
    for i, img in enumerate(imgs_retrieval):
        ax = plt.figure(dpi=256)
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1)  # set border thickness
            ax.spines[axis].set_color('black')  # set to black

        outfile = os.path.join(outpath, 'suggestfig' + str(i) + '.jpg')
        plt.savefig(outfile, bbox_inches='tight')
    plt.close()
