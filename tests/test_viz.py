import warnings
import skimage.util
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
import pytest
import h5py
import tqdm

def test_display_components():
    # original params: components, cmap='gray', headless=False
    '''
    cmap = 'gray'
    im_size = int(np.sqrt(components.shape[1]))
    plotv = components.reshape((-1, im_size, im_size))
    plotv = skimage.util.montage(plotv)

    plt.switch_backend('agg')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.imshow(plotv, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    '''
    print('not implemented')



def test_scree_plot():
    # original params: explained_variance_ratio, headless=False
    #pytest.fail('not implemented')
    print('not implemented')

def changepoint_dist():
    # original params: cps, headless=False
    #pytest.fail('not implemented')
    print('not implemented')