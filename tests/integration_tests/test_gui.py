import os
import sys
import shutil
import ruamel.yaml as yaml
from unittest import TestCase
from os.path import join, exists
from moseq2_pca.gui import (
    train_pca_command,
    apply_pca_command,
    compute_changepoints_command,
)


def _is_file(*args):
    return exists(join(*args))


class TestGUI(TestCase):

    def test_train_pca_command(self):
        data_dir = "data/"
        config_file = "data/config.yaml"
        output_dir = "data/tmp_pca"
        output_file = "pca"

        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

            config_data["use_fft"] = True
            config_data["missing_data"] = False

        with open(config_file, "w") as f:
            yaml.safe_dump(config_data, f)

        # in case it asks for user input
        stdin = "data/stdin.txt"
        with open(stdin, "w") as f:
            f.write("Y")

        progress_paths = {"train_data_dir": data_dir, "config_file": config_file}

        sys.stdin = open(stdin)

        train_pca_command(progress_paths, output_dir, output_file)

        assert exists(output_dir), "PCA path was not created."

        out_files = os.listdir(output_dir)

        assert len(out_files) >= 6, "PCA did not correctly generate all required files."
        assert _is_file(
            output_dir, "pca.h5"
        ), "PCA file was not created in the correct location"
        assert _is_file(output_dir, "pca.yaml"), "PCA metadata file is missing"
        assert _is_file(
            output_dir, "pca_components.pdf"
        ), "PCA components image is missing"
        assert _is_file(output_dir, "pca_scree.pdf"), "PCA Scree plot is missing"

        sys.stdin = open(stdin)

        train_pca_command(progress_paths, output_dir, output_file)

        shutil.rmtree(output_dir)
        os.remove(stdin)

    def test_apply_pca_command(self):
        data_dir = "data/"
        config_file = "data/config.yaml"
        output_dir = "data/tmp_pca"
        output_file = "pca"

        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

            config_data["use_fft"] = True
            config_data["missing_data"] = False

        with open(config_file, "w") as f:
            yaml.safe_dump(config_data, f)

        # in case it asks for user input
        stdin = "data/stdin.txt"
        with open(stdin, "w") as f:
            f.write("Y")

        progress_paths = {"train_data_dir": data_dir, "config_file": config_file}

        sys.stdin = open(stdin)
        train_pca_command(progress_paths, output_dir, output_file)

        data_dir = "data/"
        index_file = "data/test_index.yaml"
        config_file = "data/config.yaml"
        outpath = "data/tmp_pca"
        output_file = "pca_scores2"

        if not exists(outpath):
            os.makedirs(outpath)

        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
            config_data["pca_file"] = join(outpath, "pca.h5")

            config_data["use_fft"] = True
            config_data["missing_data"] = False

        with open(config_file, "w") as f:
            yaml.safe_dump(config_data, f)

        progress_paths = {
            "train_data_dir": data_dir,
            "config_file": config_file,
            "index_file": index_file,
            "pca_dirname": outpath,
        }

        sys.stdin = open(stdin)

        apply_pca_command(progress_paths, output_file)
        assert _is_file(outpath, output_file + ".h5"), "Scores file was not created."
        sys.stdin = open(stdin)

    def test_compute_changepoints_command(self):
        data_dir = "data/"
        config_file = "data/config.yaml"
        outpath = "data/_pca/"
        output_file = "changepoints2"

        progress_paths = {
            "train_data_dir": data_dir,
            "config_file": config_file,
            "pca_dirname": outpath,
        }

        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
            config_data["pca_file"] = join(outpath, "pca.h5")

            config_data["use_fft"] = True
            config_data["missing_data"] = False

        with open(config_file, "w") as f:
            yaml.safe_dump(config_data, f)

        compute_changepoints_command(data_dir, progress_paths, output_file)

        assert exists(outpath), "PCA Path was not found"
        assert _is_file(outpath, output_file + ".h5")
        assert _is_file(outpath, output_file + "_dist.pdf")
        assert _is_file(outpath, output_file + "_dist.png")

        os.remove(join(outpath, output_file + ".h5"))
        os.remove(join(outpath, output_file + "_dist.pdf"))
        os.remove(join(outpath, output_file + "_dist.png"))
