import os
from unittest import TestCase
from moseq2_pca.gui import train_pca_command, apply_pca_command, compute_changepoints_command

class TestGUI(TestCase):

    def test_train_pca_command(self):
        data_dir = 'data/'
        config_file = 'data/config.yaml'
        output_dir = 'data/tmp_pca'
        output_file = 'pca'

        train_pca_command(data_dir, config_file, output_dir, output_file)
        assert len(os.listdir(output_dir)) >= 6
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
        os.removedirs(output_dir)


    def test_apply_pca_command(self):
        data_dir = 'data/'
        index_file = 'data/test_index.yaml'
        config_file = 'data/config.yaml'
        outpath = '_pca'
        output_file = 'pca_scores2'

        if not os.path.exists(outpath):
            os.makedirs(outpath)

        apply_pca_command(data_dir, index_file, config_file, outpath, output_file)
        assert os.path.exists(os.path.join('data', outpath, 'pca_scores2.h5'))

        os.remove(os.path.join('data', outpath, 'pca_scores2.h5'))


    def test_compute_changepoints_command(self):
        data_dir = 'data/'
        config_file = 'data/config.yaml'
        outpath = '_pca'
        output_file = 'changepoints2'

        compute_changepoints_command(data_dir, config_file, outpath, output_file)

        assert os.path.exists(outpath)
        assert os.path.exists(os.path.join(data_dir, outpath, 'changepoints2.h5'))
        assert os.path.exists(os.path.join(data_dir, outpath, 'changepoints2_dist.pdf'))
        assert os.path.exists(os.path.join(data_dir, outpath, 'changepoints2_dist.png'))

        os.remove(os.path.join('data', outpath, 'changepoints2.h5'))
        os.remove(os.path.join('data', outpath, 'changepoints2_dist.png'))
        os.remove(os.path.join('data', outpath, 'changepoints2_dist.pdf'))
