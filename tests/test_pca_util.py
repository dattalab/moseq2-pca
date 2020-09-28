import unittest
import pytest
import os
import h5py
import numpy as np
import cv2
from copy import deepcopy
import ruamel.yaml as yaml
import platform
import pathlib
import psutil
import re
import scipy.signal
from chest import Chest
import dask.array as da
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
import time
import tqdm
import warnings
from moseq2_pca.pca.util import *


def test_mask_data():
    # original params: original data, mask, new_data
    nframes = 100

    fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
    tmp_image = np.ones((80, 80), dtype='int8')
    center = np.array(tmp_image.shape) // 2

    mouse_dims = np.array(fake_mouse.shape) // 2

    tmp_image[center[0] - mouse_dims[0]:center[0] + mouse_dims[0],
    center[1] - mouse_dims[1]:center[1] + mouse_dims[1]] = fake_mouse

    frames = np.tile(tmp_image, (nframes, 1, 1))
    mask = frames.reshape(-1, frames.shape[1] * frames.shape[2])

    new_data = np.zeros((frames.shape[1],frames.shape[2]))

    if mask.shape != (100, 6400):

        pytest.fail(f'incorrect mask shape {mask.shape}\n')

    if new_data.shape != (80,80):
        pytest.fail(f'incorrect new_data shape {new_data.shape}\n')

    frames[mask] = new_data

def test_train_pca_dask():
    # original params: dask_array, clean_params, use_fft, rank,
#                    cluster_type, client, workers,
#                    cache, mask=None, iters=10, recon_pcs=10,
#                    min_height=10, max_height=100

    #pytest.fail('not implemented')
    print('not implemented')