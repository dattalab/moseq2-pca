import os
import shutil
import ruamel.yaml as yaml
from unittest import TestCase
from os.path import join, exists
from click.testing import CliRunner
from moseq2_pca.cli import clip_scores, train_pca, apply_pca, compute_changepoints

class TestCli(TestCase):

    def test_clip_scores(self):

        data_dir = 'data/'
        h5path = join(data_dir, 'test_scores.h5')
        clip_samples = '15'

        clip_params = [h5path, clip_samples]

        runner = CliRunner()
        result = runner.invoke(clip_scores, clip_params)
        outputfile = 'data/test_scores_clip.h5'

        assert exists(outputfile), "Clipped scores file was not created"
        os.remove(outputfile)
        assert (result.exit_code == 0), "CLI function did not complete successfully"

    def test_train_pca(self):
        data_dir = 'data'
        out_dir = 'data/tmp_pca'

        train_params_local = ['-i', data_dir,
                              '--cluster-type', 'local',
                              '--local-processes', 'False',
                              '--missing-data',
                              '-o', out_dir]

        # in case it asks for user input
        stdin = 'data/stdin.txt'
        with open(stdin, 'w') as f:
            f.write('Y')

        runner = CliRunner()

        result = runner.invoke(train_pca,
                               train_params_local,
                               catch_exceptions=False)

        assert (result.exit_code == 0), "CLI Command did not successfully complete"
        assert exists(out_dir), "pca directory was not successfully created"
        outfiles = [f for f in os.listdir(out_dir)]

        assert ('pca.h5' in outfiles and 'pca.yaml' in outfiles and \
                'pca_components.pdf' in outfiles and 'pca_scree.pdf' in outfiles), \
            'PCA files were not computed successfully'

        shutil.rmtree(out_dir)
        os.remove(stdin)

    def test_apply_pca(self):
        data_dir = 'data'
        outpath = 'data/_pca'
        pca_yaml = 'data/_pca/pca.yaml'

        with open(pca_yaml, 'r') as f:
            pca_meta = yaml.safe_load(f)

        pca_meta['missing_data'] = True

        with open(pca_yaml, 'w') as f:
            yaml.safe_dump(pca_meta, f)


        apply_params_local = ['-i', data_dir,
                              '-o', outpath,
                              '--output-file', 'pca_scores1',
                              ]

        runner = CliRunner()

        result = runner.invoke(apply_pca,
                               apply_params_local,
                               catch_exceptions=False)

        assert result.exit_code == 0, "CLI command did not successfully complete"
        assert exists(outpath), "pca directory does not exist"
        assert exists(join(outpath, 'pca_scores1.h5')), "pca scores were not correctly saved"

        with open(str(pca_yaml), 'r') as f:
            pca_meta = yaml.safe_load(f)

        pca_meta['missing_data'] = False

        with open(str(pca_yaml), 'w') as f:
            yaml.safe_dump(pca_meta, f)

        os.remove(join(outpath, 'pca_scores1.h5'))

    def test_compute_changepoints(self):
        data_path = 'data'
        outpath = 'data/_pca'

        if not exists(outpath):
            os.makedirs(outpath)

        cc_params_local = ['-i', data_path, '-o', outpath,
                           '--output-file', 'changepoints1']

        runner = CliRunner()

        result = runner.invoke(compute_changepoints,
                          cc_params_local,
                          catch_exceptions=False)

        assert result.exit_code == 0, "CLI command did not successfully complete"
        assert exists(outpath), "simulated path was not generated"
        assert exists(join(outpath, 'changepoints1.h5')), "changepoints were not computed"
        assert exists(join(outpath, 'changepoints1_dist.pdf')), "changepoint pdf was not create"
        assert exists(join(outpath, 'changepoints1_dist.png')), "changepoint image was not create"

        os.remove(join(outpath, 'changepoints1.h5'))
        os.remove(join(outpath, 'changepoints1_dist.pdf'))
        os.remove(join(outpath, 'changepoints1_dist.png'))
