import warnings
import skimage.util
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode


def display_components(components, cmap='gray', headless=False):
    '''
    Creates grid of computed Principal Components.
    Parameters
    ----------
    components (np.ndarray): components to graph
    cmap (str): color map to use
    headless (bool): trim first element in PC list

    Returns
    -------
    plt (plt.figure): figure to save/graph
    ax (plt.ax): figure axis variable
    '''

    im_size = int(np.sqrt(components.shape[1]))
    plotv = components.reshape((-1, im_size, im_size))
    plotv = skimage.util.montage(plotv)

    if headless:
        plt.switch_backend('agg')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.imshow(plotv, cmap=cmap)
    plt.xticks([])
    plt.yticks([])

    return plt, ax


def scree_plot(explained_variance_ratio, headless=False):
    '''
    Creates Scree plot describing principal components.
    Parameters
    ----------
    explained_variance_ratio (np.array): explained variance ratio of each principal component
    headless (bool): trim first element in PC list

    Returns
    -------
    plt (plt.figure): figure to save/graph
    '''

    csum = np.cumsum(explained_variance_ratio)*1e2

    if headless:
        plt.switch_backend('agg')

    sns.set_style('ticks')
    plt.plot(np.cumsum(explained_variance_ratio)*1e2)

    idx = np.where(csum >= 90)
    plt.ylim((0, 100))
    plt.xlim((0, len(explained_variance_ratio)))

    if len(idx[0]) > 0:
        idx = np.min(idx)
        plt.plot([idx, idx], [0, csum[idx]], 'k-')
        plt.plot([0, idx], [csum[idx], csum[idx]], 'k-')
        plt.title('{:0.2f}% in {} pcs'.format(csum[idx], idx + 1))

    plt.ylabel('Variance explained (percent)')
    plt.xlabel('nPCs')
    sns.despine()

    return plt


def changepoint_dist(cps, headless=False):
    '''
    Creates bar plot describing computed Changepoint Distribution.
    Parameters
    ----------
    cps (np.ndarray): changepoints to graph
    headless (bool): trim first element in PC list

    Returns
    -------
    plt (plt.figure): figure to save/graph
    ax (plt.ax): figure axis variable
    '''
    if cps.size > 0:

        if headless:
            plt.switch_backend('agg')

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        sns.set_style('ticks')

        ax = sns.distplot(cps, kde_kws={'gridsize': 600}, bins=np.linspace(0, 10, 100))
        ax.set_xlim((0, 2))
        ax.set_xticks(np.linspace(0, 2, 11))

        s = "Mean, median, mode (s) = {0}, {1}, {2}".format(str(np.mean(cps)), str(np.median(cps)), str(mode(cps)[0][0][0]))
        plt.text(.5, 2, s, fontsize=6)
        plt.ylabel('P(block duration)')
        plt.xlabel('Block duration (s)')
        sns.despine()

        return plt, ax
    else:
        warnings.warn('No changepoints detected - check if mouse is present or moving')
