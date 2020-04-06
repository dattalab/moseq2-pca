import cv2
from unittest import TestCase
from moseq2_pca.pca.util import *
from tempfile import TemporaryDirectory, NamedTemporaryFile

class TestPCAUtils(TestCase):
    def test_mask_data(self):
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

        new_data = np.zeros((frames.shape[1], frames.shape[2]))

        assert(mask.shape == (100, 6400))

        assert(new_data.shape == (80, 80))


        frames[mask] = new_data

    def test_train_pca_dask(self):
        # original params: dask_array, clean_params, use_fft, rank,
    #                    cluster_type, client, workers,
    #                    cache, mask=None, iters=10, recon_pcs=10,
    #                    min_height=10, max_height=100

        #pytest.fail('not implemented')
        print('not implemented')