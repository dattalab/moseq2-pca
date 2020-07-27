import os
import cv2
import h5py
import pytest
import numpy as np
import scipy.signal
import ruamel.yaml as yaml
from unittest import TestCase
from dask.distributed import Client, LocalCluster
from moseq2_pca.util import gaussian_kernel1d, gauss_smooth, read_yaml, insert_nans, \
    recursive_find_h5s, clean_frames, select_strel, get_timestamp_path, \
    get_metadata_path, initialize_dask, get_rps, get_changepoints, get_timestamps


class TestUtils(TestCase):

    def test_recursive_find_h5s(self):
        # original params: root_dir=os.getcwd(), ext='.h5', yaml_string='{}.yaml'
        input_dir = 'data/'
        h5s, dicts, yamls = recursive_find_h5s(input_dir)

        print(h5s)
        print(dicts)
        print(yamls)
        assert len(h5s) == len(dicts) == len(yamls)

        input_dir1 = 'data/proc/'
        h5s1, dicts1, yamls1 = recursive_find_h5s(input_dir1)
        assert len(h5s1) == len(dicts1) == len(yamls1)
        assert len(h5s) == len(h5s1)

        input_dir2 = 'data/_pca/'
        h5s2, dicts2, yamls2 = recursive_find_h5s(input_dir2)
        assert len(h5s2) == len(dicts2) == len(yamls2)
        assert len(h5s1) != len(h5s2)

    def test_gauss_smooth(self):
        # original params: signal, win_length=None, sig=1.5, kernel=None
        sig = 1.5
        win_length = None
        kernel = None
        if kernel is None:
            kernel = gaussian_kernel1d(n=win_length, sig=sig)

        truth_result = scipy.signal.convolve([sig], kernel, mode='same', method='direct')
        self.assertListEqual(list(truth_result), [0.3194797971506902])

        test_result1 = gauss_smooth([sig], win_length=win_length, kernel=kernel, sig=sig)
        assert truth_result == test_result1

        test_result2 = gauss_smooth([sig], win_length=None, kernel=None, sig=sig)
        assert truth_result == test_result2

    def test_gaussian_kernel1d(self):
        # original params: n = None, sig=3
        n = None
        sig = 3

        if n is None:
            n = np.ceil(sig * 4)
            assert (n == 12)

        points = np.arange(-n, n)

        kernel = np.exp(-(points ** 2.0) / (2.0 * sig ** 2.0))
        kernel /= np.sum(kernel)

        mock_res = [4.46133330e-05, 1.60101908e-04, 5.14130542e-04, 1.47739069e-03,
                    3.79893942e-03, 8.74126801e-03, 1.79983031e-02, 3.31614678e-02,
                    5.46740173e-02, 8.06627984e-02, 1.06490445e-01, 1.25803596e-01,
                    1.32990471e-01, 1.25803596e-01, 1.06490445e-01, 8.06627984e-02,
                    5.46740173e-02, 3.31614678e-02, 1.79983031e-02, 8.74126801e-03,
                    3.79893942e-03, 1.47739069e-03, 5.14130542e-04, 1.60101908e-04]

        pytest.approx(all([a == b for a, b in zip(mock_res, kernel)]))

        test_kernel = gaussian_kernel1d(n, sig)

        pytest.approx(all([a == b for a, b in zip(test_kernel, kernel)]))

    def test_clean_frames(self):
        nframes = 20

        fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
        tmp_image = np.zeros((80, 80), dtype='int8')
        center = np.array(tmp_image.shape) // 2

        mouse_dims = np.array(fake_mouse.shape) // 2

        tmp_image[center[0] - mouse_dims[0]:center[0] + mouse_dims[0],
        center[1] - mouse_dims[1]:center[1] + mouse_dims[1]] = fake_mouse

        frames = np.tile(tmp_image, (nframes, 1, 1))

        medfilter_space = [1, 1]
        gaussfilter_space = None
        medfilter_time = [3]
        gaussfilter_time = None
        detrend_time = 1
        tailfilter = None

        test_output = clean_frames(frames, medfilter_space=medfilter_space, gaussfilter_space=gaussfilter_space,
                     medfilter_time=medfilter_time, gaussfilter_time=gaussfilter_time, detrend_time=detrend_time,
                     tailfilter=tailfilter, tail_threshold=5)

        np.testing.assert_equal(np.any(np.not_equal(frames, test_output)), True)

    def test_select_strel(self):
        # original params: string='e', size=(10,10)
        string0 = ''
        string1 = 'e'
        string2 = 'r'
        size = (10, 10)
        strel = None
        mock_strel0 = None
        mock_strel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
        mock_strel2 = cv2.getStructuringElement(cv2.MORPH_RECT, size)

        test0 = select_strel(strel, size)
        test01 = select_strel(string0, size)
        test1 = select_strel(string1, size)
        test2 = select_strel(string2, size)

        assert test0 == test01 == mock_strel0
        assert test1.all() == mock_strel1.all()
        assert test2.all() == mock_strel2.all()

    def test_read_yaml(self):
        # original param: yaml_file
        yaml_file = 'data/config.yaml'
        try:
            with open(yaml_file, 'r') as f:
                dat = f.read()
                try:
                    truth_dict = yaml.safe_load(dat)
                except yaml.constructor.ConstructorError:
                    truth_dict = yaml.safe_load(dat)
                    pytest.fail('yaml exception thrown')
        except IOError:
            truth_dict = {}
            pytest.fail('IOERROR')

        if truth_dict == {}:
            if dat != None:
                pytest.fail('no data read.')

        test_dict = read_yaml(yaml_file)
        assert test_dict == truth_dict

    def test_get_timestamps(self):
        h5file = ['data/proc/results_00.h5']
        with pytest.warns(None) as record:
            get_timestamps(h5file)
        assert not record # no warnings emitted

    def test_get_timestamp_path(self):
        # original param: h5file path
        h5file = 'data/proc/results_00.h5'
        test_path = get_timestamp_path(h5file)

        true_path = []
        with h5py.File(h5file, 'r') as f:
            if '/timestamps' in f:
                true_path.append('/timestamps')
            elif '/metadata/timestamps' in f:
                true_path.append('/metadata/timestamps')

        assert test_path in true_path

    def test_get_metadata_path(self):
        # original param: h5file path
        h5file = 'data/proc/results_00.h5'
        test_path = get_metadata_path(h5file)

        true_path = []
        with h5py.File(h5file, 'r') as f:
            if '/metadata/acquisition' in f:
                true_path.append('/metadata/acquisition')
            elif '/metadata/extraction' in f:
                true_path.append('/metadata/extraction')

        assert test_path in true_path

    # TODO: possibly implement some kwargs edge cases
    def test_initialize_dask(self):

        nworkers = 50
        processes = 1
        memory = '4GB'
        cores = 1
        wall_time = '01:00:00'
        queue = 'debug'
        cluster_type = 'local'
        timeout = 10
        cache_path = os.path.expanduser('~/moseq2_pca')

        client, cluster, workers = initialize_dask(nworkers=nworkers, processes=processes, memory=memory,
                                                   cores=cores, wall_time=wall_time, queue=queue,
                                                   cluster_type=cluster_type,
                                                   timeout=timeout, cache_path=cache_path)

        assert isinstance(client, Client)
        assert isinstance(cluster, LocalCluster)
        assert isinstance(workers, dict)
        client.close()
        cluster.close()

    def test_get_rps(self):
        rps = 600
        nframes = 20

        fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
        tmp_image = np.zeros((80, 80), dtype='int8')
        center = np.array(tmp_image.shape) // 2

        mouse_dims = np.array(fake_mouse.shape) // 2

        tmp_image[center[0] - mouse_dims[0]:center[0] + mouse_dims[0],
        center[1] - mouse_dims[1]:center[1] + mouse_dims[1]] = fake_mouse

        frames = np.tile(tmp_image, (nframes, 1, 1))

        if frames.ndim == 3:
            use_frames = frames.reshape(-1, np.prod(frames.shape[1:]))
        elif frames.ndim == 2:
            use_frames = frames

        truth_rproj = use_frames.dot(np.random.randn(use_frames.shape[1], rps).astype('float32'))

        if truth_rproj.shape != (nframes, rps):
            pytest.fail('incorrect shape')

        # normalizing
        truth_norm_rproj = scipy.stats.zscore(scipy.stats.zscore(truth_rproj).T)

        if truth_norm_rproj.shape != (rps, nframes):
            pytest.fail('incorrect shape most normalize-transpose')

        test_norm_rproj = get_rps(frames, rps=rps, normalize=True)

        assert truth_norm_rproj.all() == test_norm_rproj.all()

        test_rproj = get_rps(frames, rps=rps, normalize=False)
        assert test_rproj.all() == truth_rproj.all()

    def test_get_changepoints(self):

        input_dir = 'data/'
        pca_scores = 'data/test_scores.h5'

        h5s, dicts, yamls = recursive_find_h5s(input_dir)

        for h5, yml in zip(h5s, yamls):
            data = read_yaml(yml)
            uuid = data['uuid']
            with h5py.File(pca_scores, 'r') as f:
                scores = f['scores/{}'.format(uuid)]
                scores_idx = f['scores_idx/{}'.format(uuid)]
                scores = scores[~np.isnan(scores_idx), :]

        k = 5
        sigma = 3
        peak_height = .5
        peak_neighbors = 1
        baseline = True
        timestamps = None

        cps, normed_df = get_changepoints(scores, k=k,
                         sigma=sigma,
                         peak_height=peak_height,
                         peak_neighbors=peak_neighbors,
                         baseline=baseline,
                         timestamps=timestamps)

        np.testing.assert_equal(cps, [[25], [27]])
        assert isinstance(normed_df, np.ndarray)
        assert len(normed_df) > 0


    def test_insert_nans(self):
        input_dir = 'data/'
        pca_scores = 'data/test_scores.h5'

        h5s, dicts, yamls = recursive_find_h5s(input_dir)

        for h5, yml in zip(h5s, yamls):
            data = read_yaml(yml)
            uuid = data['uuid']
            with h5py.File(pca_scores, 'r') as f:
                truth_scores = f['scores/{}'.format(uuid)][()]
                truth_scores_idx = f['scores_idx/{}'.format(uuid)][()]
                truth_scores = truth_scores[~np.isnan(truth_scores_idx), :]

        h5file = 'data/proc/results_00.h5'
        ts_path = get_timestamp_path(h5file)
        with h5py.File(h5file, 'r') as f:
            timestamps = f[ts_path][...] / 1000.0

        test_scores, test_score_idx, filled_timestamps = insert_nans(data=truth_scores, timestamps=timestamps,
                                           fps=np.round(1 / np.mean(np.diff(timestamps))).astype('int'))

        assert len(timestamps) < len(filled_timestamps)
        assert len(truth_scores) < len(test_scores)
        assert truth_scores_idx.all() == test_score_idx.all()