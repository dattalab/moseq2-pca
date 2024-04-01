import os
import shutil
import ruamel.yaml as yaml
from unittest import TestCase
from os.path import join, exists
from click.testing import CliRunner
from moseq2_pca.cli import clip_scores, train_pca, apply_pca, compute_changepoints


def run_pca(data_dir, out_dir):

    train_params_local = [
        "-i",
        data_dir,
        "--cluster-type",
        "local",
        "--local-processes",
        "False",
        "--gaussfilter-time",
        "3",
        "--missing-data",
        "--missing-data-iters",
        1,
        "--dask-port",
        ":1234",
        "--config-file",
        "data/config.yaml",
        "-o",
        out_dir,
    ]

    # in case it asks for user input
    stdin = "data/stdin.txt"
    with open(stdin, "w") as f:
        f.write("Y")

    runner = CliRunner()

    result = runner.invoke(train_pca, train_params_local, catch_exceptions=False)

    assert result.exit_code == 0, "CLI Command did not successfully complete"
    assert exists(out_dir), "pca directory was not successfully created"
    outfiles = [f for f in os.listdir(out_dir)]

    assert (
        "pca.h5" in outfiles
        and "pca.yaml" in outfiles
        and "pca_components.pdf" in outfiles
        and "pca_scree.pdf" in outfiles
    ), "PCA files were not computed successfully"

    os.remove(stdin)


def run_apply(data_dir, out_dir):
    # in case it asks for user input
    stdin = "data/stdin.txt"
    with open(stdin, "w") as f:
        f.write("Y")

    apply_params_local = [
        "-i",
        data_dir,
        "-o",
        out_dir,
        "--output-file",
        "pca_scores1",
        "--cluster-type",
        "nodask",
        "--verbose",
    ]

    print("moseq2-pca apply-pca", " ".join(apply_params_local))

    runner = CliRunner()

    result = runner.invoke(apply_pca, apply_params_local, catch_exceptions=False)

    assert result.exit_code == 0, "CLI command did not successfully complete"
    assert exists(out_dir), "pca directory does not exist"
    assert exists(
        join(out_dir, "pca_scores1.h5")
    ), "pca scores were not correctly saved"


class TestCli(TestCase):

    def test_clip_scores(self):

        data_dir = "data/"
        h5path = join(data_dir, "test_scores.h5")
        clip_samples = "15"

        clip_params = [h5path, clip_samples]

        runner = CliRunner()
        result = runner.invoke(clip_scores, clip_params)
        outputfile = "data/test_scores_clip.h5"

        assert exists(outputfile), "Clipped scores file was not created"
        assert result.exit_code == 0, "CLI function did not complete successfully"

    def test_train_pca(self):

        data_dir = "data/"
        out_dir = "data/tmp_pca"

        run_pca(data_dir, out_dir)
        shutil.rmtree(out_dir)

    def test_apply_pca(self):
        data_dir = "data/"
        out_dir = "data/tmp_pca"

        run_pca(data_dir, out_dir)
        run_apply(data_dir, out_dir)

        shutil.rmtree(out_dir)

    def test_compute_changepoints(self):
        data_dir = "data/"
        out_dir = "data/tmp_pca"

        run_pca(data_dir, out_dir)
        run_apply(data_dir, out_dir)

        pca_yaml = "data/tmp_pca/pca.yaml"
        config = "data/config.yaml"

        with open(config, "r") as f:
            config_data = yaml.safe_load(f)

        config_data["pca_file"] = None
        config_data["pca_file_scores"] = None

        with open(config, "w") as f:
            yaml.safe_dump(config_data, f)

        os.rename("data/tmp_pca/pca_scores1.h5", "data/tmp_pca/pca_scores.h5")

        with open(pca_yaml, "r") as f:
            pca_meta = yaml.safe_load(f)

        pca_meta["missing_data"] = True

        with open(pca_yaml, "w") as f:
            yaml.safe_dump(pca_meta, f)

        cc_params_local = [
            "-i",
            data_dir,
            "-o",
            out_dir,
            "--config-file",
            config,
            "--pca-file-scores",
            "data/tmp_pca/pca_scores.h5",
            "--output-file",
            "changepoints1",
            "-v",
        ]

        runner = CliRunner()

        result = runner.invoke(
            compute_changepoints, cc_params_local, catch_exceptions=False
        )

        assert result.exit_code == 0, "CLI command did not successfully complete"
        assert exists(out_dir), "simulated path was not generated"
        assert exists(
            join(out_dir, "changepoints1.h5")
        ), "changepoints were not computed"
        assert exists(
            join(out_dir, "changepoints1_dist.pdf")
        ), "changepoint pdf was not create"
        assert exists(
            join(out_dir, "changepoints1_dist.png")
        ), "changepoint image was not create"

        shutil.rmtree(out_dir)
