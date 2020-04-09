import os
from unittest import TestCase
from click.testing import CliRunner
from moseq2_pca.cli import clip_scores, train_pca, apply_pca, compute_changepoints


class TestCli(TestCase):

    def test_clip_scores(self):

        data_dir = 'data/'
        h5path = os.path.join(data_dir, 'test_scores.h5')
        clip_samples = '15'

        clip_params = [h5path, clip_samples]

        runner = CliRunner()
        result = runner.invoke(clip_scores, clip_params)
        outputfile = 'data/test_scores_clip.h5'

        assert(os.path.exists(outputfile))
        os.remove(outputfile)
        assert (result.exit_code == 0)

    def test_train_pca(self):
        data_dir = 'data/'

        train_params_local = ['-i', data_dir,
                        '--cluster-type', 'local',
                        '-o', os.path.join(data_dir, 'tmp_pca')]


        runner = CliRunner()

        result = runner.invoke(train_pca,
                               train_params_local,
                               catch_exceptions=False)

        assert (os.path.exists(os.path.join(data_dir, 'tmp_pca') == True))
        outfiles = os.listdir(os.path.join(data_dir, 'tmp_pca'))
        assert ('pca.h5' in outfiles and 'pca.yaml' in outfiles and \
                'pca_components.pdf' in outfiles and 'pca_scree.pdf' in outfiles)

        for file in os.listdir(os.path.join(data_dir, 'tmp_pca')):
            os.remove(os.path.join(data_dir, 'tmp_pca', file))
        os.removedirs(os.path.join(data_dir, 'tmp_pca'))

        assert(result.exit_code == 0)


    def test_apply_pca(self):
        data_dir = 'data/_pca/'

        outpath = 'tmp_pca'
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        apply_params_local = ['-i', data_dir,
                              '-o', outpath,
                              '--output-file', 'pca_scores1',
                              ]

        runner = CliRunner()

        result = runner.invoke(apply_pca,
                          apply_params_local,
                          catch_exceptions=False)

        assert os.path.exists(outpath)
        assert os.path.exists(os.path.join('data', outpath, 'pca_scores1.h5'))
        assert result.exit_code == 0

        for file in os.listdir(os.path.join('data', outpath)):
            os.remove(os.path.join('data', outpath, file))
        os.removedirs(os.path.join('data', outpath))



    def test_compute_changepoints(self):
        data_path = 'data/_pca/'
        outpath = 'tmp_pca'
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        cc_params_local = ['-i', data_path,
                           '-o', outpath,
                           '--output-file', 'changepoints1']

        runner = CliRunner()

        result = runner.invoke(compute_changepoints,
                          cc_params_local,
                          catch_exceptions=False)

        assert os.path.exists(outpath)
        assert os.path.exists(os.path.join('data', outpath, 'changepoints1.h5'))
        assert result.exit_code == 0

        for file in os.listdir(os.path.join('data', outpath)):
            os.remove(os.path.join('data', outpath, file))
        os.removedirs(os.path.join('data', outpath))

