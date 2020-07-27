import os
import cv2
import h5py
import numpy as np
import dask.array as da
import ruamel.yaml as yaml
from unittest import TestCase
from dask.distributed import Client
from moseq2_pca.util import recursive_find_h5s
from moseq2_pca.helpers.data import get_pca_yaml_data
from moseq2_pca.pca.util import mask_data, train_pca_dask, apply_pca_dask, apply_pca_local, get_changepoints_dask

class TestPCAUtils(TestCase):

    def test_mask_data(self):
        nframes = 10

        fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
        tmp_image = np.ones((80, 80), dtype='int8')
        center = np.array(tmp_image.shape) // 2

        mouse_dims = np.array(fake_mouse.shape) // 2

        tmp_image[center[0] - mouse_dims[0]:center[0] + mouse_dims[0],
        center[1] - mouse_dims[1]:center[1] + mouse_dims[1]] = fake_mouse

        frames = np.tile(tmp_image, (nframes, 1, 1))
        init_frames = frames.copy()
        mask = frames.reshape(-1, frames.shape[1] * frames.shape[2])

        new_data = np.zeros(frames.shape)

        assert(mask.shape == (nframes, 6400))
        assert(new_data.shape == (nframes, 80, 80))

        test_out = mask_data(frames, mask, new_data)
        frames[mask] = new_data[mask]

        assert (frames.any() > 0) == (init_frames.all() == 0)
        assert frames.all() == test_out.all()

    def test_train_pca_dask(self):

        input_dir = 'data/proc/'
        config_file = 'data/config.yaml'

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        h5s, dicts, yamls = recursive_find_h5s(input_dir)
        dsets = [h5py.File(h5, mode='r')['/frames'] for h5 in h5s]

        arrays = [da.from_array(dset, chunks=(1000, -1, -1)) for dset in dsets]
        stacked_array = da.concatenate(arrays, axis=0)

        stacked_array[stacked_array < 10] = 0
        stacked_array[stacked_array > 100] = 0
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(config_data['tailfilter_size']))
        tailfilter = strel

        clean_params = {
            'gaussfilter_space': config_data['gaussfilter_space'],
            'gaussfilter_time': config_data['gaussfilter_time'],
            'tailfilter': tailfilter,
            'medfilter_time': config_data['medfilter_time'],
            'medfilter_space': config_data['medfilter_space']
        }

        client = Client(processes=True)

        output_dict = \
            train_pca_dask(dask_array=stacked_array, mask=None,
                           clean_params=clean_params, use_fft=config_data['use_fft'],
                           rank=config_data['rank'], cluster_type=config_data['cluster_type'],
                           min_height=config_data['min_height'],
                           max_height=config_data['max_height'], client=client,
                           iters=config_data['missing_data_iters'],
                           recon_pcs=config_data['recon_pcs'])
        client.restart()
        client.close()

        assert 'components' in output_dict.keys()
        assert 'singular_values' in output_dict.keys()
        assert 'explained_variance' in output_dict.keys()
        assert 'explained_variance_ratio' in output_dict.keys()
        assert 'mean' in output_dict.keys()


    def test_apply_pca_local(self):

        input_dir = 'data/proc/'
        pca_path = 'data/_pca/pca'
        save_file = 'data/_pca/local_test_pca_scores'
        config_file = 'data/config.yaml'

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        with h5py.File(f'{pca_path}.h5', 'r') as f:
            pca_components = f['components'][()]

        use_fft, clean_params, mask_params, missing_data = get_pca_yaml_data(f'{pca_path}.yaml')

        h5s, dicts, yamls = recursive_find_h5s(input_dir)

        apply_pca_local(pca_components=pca_components, h5s=h5s, yamls=yamls,
                        use_fft=use_fft, clean_params=clean_params,
                        save_file=save_file, chunk_size=config_data['chunk_size'],
                        mask_params=mask_params, fps=config_data['fps'],
                        missing_data=missing_data)

        assert os.path.exists(f'{save_file}.h5')
        os.remove(f'{save_file}.h5')


    def test_apply_pca_dask(self):

        input_dir = 'data/proc/'
        pca_path = 'data/_pca/pca'
        save_file = 'data/_pca/dask_test_pca_scores'

        with h5py.File(f'{pca_path}.h5', 'r') as f:
            pca_components = f['components'][()]

        use_fft, clean_params, mask_params, missing_data = get_pca_yaml_data(f'{pca_path}.yaml')

        h5s, dicts, yamls = recursive_find_h5s(input_dir)

        chunk_size = 100
        client = Client(processes=True)

        apply_pca_dask(pca_components, h5s, yamls, use_fft, clean_params,
                       save_file, chunk_size, mask_params, missing_data,
                       client)

        client.restart()

        assert os.path.exists(f'{save_file}.h5')
        os.remove(f'{save_file}.h5')

        missing_data = True

        apply_pca_dask(pca_components, h5s, yamls, use_fft, clean_params,
                       save_file, chunk_size, mask_params, missing_data,
                       client)

        client.restart()
        client.close()

        assert os.path.exists(f'{save_file}.h5')
        os.remove(f'{save_file}.h5')

    def test_get_changepoints_dask(self):

        input_dir = 'data/proc/'
        pca_path = 'data/_pca/pca'
        save_file = 'data/_pca/test_changepoints'
        config_file = 'data/config.yaml'

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        with h5py.File(f'{pca_path}.h5', 'r') as f:
            pca_components = f['components'][()]

        use_fft, clean_params, mask_params, missing_data = get_pca_yaml_data(f'{pca_path}.yaml')

        missing_data = False

        h5s, dicts, yamls = recursive_find_h5s(input_dir)

        changepoint_params = {
            'k': config_data['klags'],
            'sigma': config_data['sigma'],
            'peak_height': config_data['threshold'],
            'peak_neighbors': config_data['neighbors'],
            'rps': config_data['dims']
        }

        chunk_size = 100
        client = Client(processes=True)

        get_changepoints_dask(changepoint_params, pca_components, h5s, yamls,
                              save_file, chunk_size, mask_params, missing_data,
                              client)
        client.restart()
        client.close()
        client = Client(processes=True)

        assert os.path.exists(f'{save_file}.h5')
        os.remove(f'{save_file}.h5')

        missing_data_save_file = 'data/_pca/dask_test_pca_scores'

        missing_data = True

        apply_pca_dask(pca_components, h5s, yamls, use_fft, clean_params,
                       missing_data_save_file, chunk_size, mask_params, missing_data,
                       client)

        assert os.path.exists(f'{missing_data_save_file}.h5')

        changepoint_params = {
            'k': config_data['klags'],
            'sigma': config_data['sigma'],
            'peak_height': config_data['threshold'],
            'peak_neighbors': config_data['neighbors'],
            'rps': config_data['dims']
        }

        get_changepoints_dask(changepoint_params, pca_components, h5s, yamls,
                              save_file, chunk_size, mask_params, missing_data,
                              client, 30, pca_scores=f'{missing_data_save_file}.h5')
        client.restart()
        client.close()

        assert os.path.exists(f'{save_file}.h5')
        assert os.path.exists(f'{missing_data_save_file}.h5')
        os.remove(f'{save_file}.h5')
        os.remove(f'{missing_data_save_file}.h5')