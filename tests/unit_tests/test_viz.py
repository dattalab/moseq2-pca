import os
import h5py
import numpy as np
from unittest import TestCase
from moseq2_pca.util import h5_to_dict
from moseq2_pca.viz import (
    display_components,
    scree_plot,
    changepoint_dist,
    plot_pca_results,
)


class TestViz(TestCase):

    def test_plot_pca_results(self):
        pca_path = "data/_pca/pca.h5"
        with h5py.File(pca_path, "r") as f:
            components = f["components"][()]
            explained_variance_ratio = f["explained_variance_ratio"][()]

        in_dict = {
            "components": components,
            "explained_variance_ratio": explained_variance_ratio,
        }
        save_file = "data/pca"
        output_dir = "data/"

        plot_pca_results(in_dict, save_file, output_dir)

        assert os.path.exists("data/pca_components.png")
        assert os.path.exists("data/pca_components.pdf")

        os.remove("data/pca_components.png")
        os.remove("data/pca_components.pdf")

        assert os.path.exists("data/pca_scree.png")
        assert os.path.exists("data/pca_scree.pdf")

        os.remove("data/pca_scree.png")
        os.remove("data/pca_scree.pdf")

    def test_display_components(self):
        # get components
        pca_path = "data/_pca/pca.h5"
        with h5py.File(pca_path, "r") as f:
            components = f["components"][()]
            plt, ax = display_components(components)
            assert plt is not None and ax is not None

    def test_scree_plot(self):
        # get explained_variance_ratio
        pca_path = "data/_pca/pca.h5"
        with h5py.File(pca_path, "r") as f:
            components = f["explained_variance_ratio"][()]
            plt = scree_plot(components)
            assert plt is not None

    def test_changepoint_dist(self):
        # original params: cps, headless=False
        save_file = "data/_pca/changepoints"

        with h5py.File(f"{save_file}.h5", "r") as f:
            cps = h5_to_dict(f, "cps")
        block_durs = np.concatenate([np.diff(cp, axis=0) for k, cp in cps.items()])

        assert block_durs.shape == (100, 1)

        out = changepoint_dist(block_durs, headless=True)

        assert out is not None
