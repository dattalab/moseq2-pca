import os
import cv2
import h5py
import pytest
import uuid
import tempfile
import numpy as np
import scipy.signal
import ruamel.yaml as yaml
from unittest import TestCase
from dask.distributed import Client, LocalCluster
from moseq2_pca.util import gaussian_kernel1d, gauss_smooth, read_yaml, insert_nans, \
    check_timestamps, recursive_find_h5s, clean_frames, select_strel, \
    get_timestamp_path, get_metadata_path, initialize_dask, get_rps, get_changepoints, h5_to_dict, \
    generate_pca_input_dir_metadata, save_pca_input_data


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
        test3 = select_strel('default', size)

        assert test0 == test01 == mock_strel0
        assert test1.all() == mock_strel1.all()
        assert test2.all() == mock_strel2.all()
        assert test3.all() == mock_strel1.all()

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
            if dat is not None:
                pytest.fail('no data read.')

        test_dict = read_yaml(yaml_file)
        assert test_dict == truth_dict

    def test_check_timestamps(self):
        h5file = ['data/proc/results_00.h5']
        with pytest.warns(None) as record:
            check_timestamps(h5file)
        assert not record  # no warnings emitted

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

    def test_h5_to_dict(self):

        h5path = 'data/test_scores.h5'
        path = 'scores/'

        test = h5_to_dict(h5path, path)

        assert isinstance(test, dict)
        assert list(test.keys()) == ['5c72bf30-9596-4d4d-ae38-db9a7a28e912', 'abe92017-1d40-495e-95ef-e420b7f0f4b9']
        assert test['5c72bf30-9596-4d4d-ae38-db9a7a28e912'].shape == (908, 50)

    def test_generate_pca_input_dir_metadata_basic(self):
        """Test basic functionality with typical parameters."""
        n_frames = 1000
        FPS = 30
        
        result = generate_pca_input_dir_metadata(n_frames, FPS)
        
        # Check structure
        assert isinstance(result, dict)
        assert 'UUID' in result
        assert 'timestamps' in result
        
        # Check UUID format
        assert isinstance(result['UUID'], str)
        # Verify it's a valid UUID format
        uuid.UUID(result['UUID'])  # Will raise ValueError if invalid
        
        # Check timestamps
        assert isinstance(result['timestamps'], np.ndarray)
        assert len(result['timestamps']) == n_frames
        assert result['timestamps'][0] == 0.0
        np.testing.assert_almost_equal(result['timestamps'][-1], n_frames / FPS)
        
        # Check timestamps are evenly spaced
        expected_timestamps = np.linspace(0, n_frames / FPS, n_frames)
        np.testing.assert_array_almost_equal(result['timestamps'], expected_timestamps)

    def test_generate_pca_input_dir_metadata_with_provided_uuid(self):
        """Test with user-provided UUID."""
        n_frames = 100
        FPS = 25
        test_uuid = "12345678-1234-5678-9abc-123456789abc"
        
        result = generate_pca_input_dir_metadata(n_frames, FPS, UUID=test_uuid)
        
        assert result['UUID'] == test_uuid
        assert len(result['timestamps']) == n_frames

    def test_generate_pca_input_dir_metadata_zero_frames(self):
        """Test edge case with zero frames."""
        n_frames = 0
        FPS = 30
        
        result = generate_pca_input_dir_metadata(n_frames, FPS)
        
        assert isinstance(result, dict)
        assert 'UUID' in result
        assert 'timestamps' in result
        assert isinstance(result['timestamps'], np.ndarray)
        assert len(result['timestamps']) == 0

    def test_generate_pca_input_dir_metadata_one_frame(self):
        """Test edge case with single frame."""
        n_frames = 1
        FPS = 30
        
        result = generate_pca_input_dir_metadata(n_frames, FPS)
        
        assert len(result['timestamps']) == 1
        assert result['timestamps'][0] == 0.0

    def test_generate_pca_input_dir_metadata_different_fps(self):
        """Test with different FPS values."""
        n_frames = 60
        
        # Test with FPS = 60
        result_60fps = generate_pca_input_dir_metadata(n_frames, 60)
        assert result_60fps['timestamps'][-1] == 1.0  # 60 frames / 60 FPS = 1 second
        
        # Test with FPS = 15  
        result_15fps = generate_pca_input_dir_metadata(n_frames, 15)
        assert result_15fps['timestamps'][-1] == 4.0  # 60 frames / 15 FPS = 4 seconds
        
        # Test with FPS = 1
        result_1fps = generate_pca_input_dir_metadata(n_frames, 1)
        assert result_1fps['timestamps'][-1] == 60.0  # 60 frames / 1 FPS = 60 seconds

    def test_generate_pca_input_dir_metadata_unique_uuids(self):
        """Test that generated UUIDs are unique across calls."""
        result1 = generate_pca_input_dir_metadata(100, 30)
        result2 = generate_pca_input_dir_metadata(100, 30)
        
        assert result1['UUID'] != result2['UUID']
        
        # Both should be valid UUIDs
        uuid.UUID(result1['UUID'])
        uuid.UUID(result2['UUID'])

    def test_generate_pca_input_dir_metadata_large_values(self):
        """Test with large frame counts."""
        n_frames = 100000
        FPS = 30
        
        result = generate_pca_input_dir_metadata(n_frames, FPS)
        
        assert len(result['timestamps']) == n_frames
        assert result['timestamps'][0] == 0.0
        np.testing.assert_almost_equal(result['timestamps'][-1], n_frames / FPS)

    def test_generate_pca_input_dir_metadata_invalid_inputs(self):
        """Test behavior with invalid inputs."""
        # Test negative frames - should still work (edge case)
        result_neg = generate_pca_input_dir_metadata(-10, 30)
        assert len(result_neg['timestamps']) == 0  # Should return empty array
        
        # Test with very high FPS
        result_high_fps = generate_pca_input_dir_metadata(1000, 10000)
        assert len(result_high_fps['timestamps']) == 1000
        assert result_high_fps['timestamps'][-1] == 0.1  # 1000/10000 = 0.1 seconds

    def test_save_pca_input_data_basic(self):
        """Test basic functionality of save_pca_input_data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup test data
            UUID = "test-uuid-12345"
            timestamps = np.linspace(0, 1, 100)  # 100 frames over 1 second
            frames = [np.random.rand(100, 80, 80).astype('float32')]  # Single batch of 100 frames
            base_path = os.path.join(temp_dir, "test_session")
            
            # Save data
            h5_path, yaml_path = save_pca_input_data(UUID, timestamps, frames, base_path)
            
            # Verify file paths
            assert h5_path == f"{base_path}.h5"
            assert yaml_path == f"{base_path}.yaml"
            assert os.path.exists(h5_path)
            assert os.path.exists(yaml_path)
            
            # Verify YAML content
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            assert yaml_data['uuid'] == UUID
            
            # Verify H5 content
            with h5py.File(h5_path, 'r') as f:
                # Check datasets exist
                assert 'frames' in f
                assert 'timestamps' in f
                
                # Check frames data
                saved_frames = f['frames'][()]
                assert saved_frames.shape == (100, 80, 80)
                assert saved_frames.dtype == np.float32
                np.testing.assert_array_almost_equal(saved_frames, frames[0])
                
                # Check timestamps data (should be in milliseconds)
                saved_timestamps = f['timestamps'][()]
                expected_timestamps = timestamps * 1000.0
                np.testing.assert_array_almost_equal(saved_timestamps, expected_timestamps)

    def test_save_pca_input_data_streaming(self):
        """Test streaming functionality with multiple frame batches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            UUID = "streaming-test-uuid"
            timestamps = np.linspace(0, 2, 200)  # 200 frames total
            
            # Create multiple batches of different sizes
            frames = [
                np.random.rand(50, 64, 64).astype('float32'),   # First batch: 50 frames
                np.random.rand(100, 64, 64).astype('float32'),  # Second batch: 100 frames  
                np.random.rand(50, 64, 64).astype('float32'),   # Third batch: 50 frames
            ]
            base_path = os.path.join(temp_dir, "streaming_session")
            
            # Save data
            h5_path, yaml_path = save_pca_input_data(UUID, timestamps, frames, base_path)
            
            # Verify the combined data
            with h5py.File(h5_path, 'r') as f:
                saved_frames = f['frames'][()]
                assert saved_frames.shape == (200, 64, 64)
                
                # Verify each batch was concatenated correctly
                np.testing.assert_array_equal(saved_frames[:50], frames[0])
                np.testing.assert_array_equal(saved_frames[50:150], frames[1])
                np.testing.assert_array_equal(saved_frames[150:], frames[2])

    def test_save_pca_input_data_single_frame(self):
        """Test with single frame."""
        with tempfile.TemporaryDirectory() as temp_dir:
            UUID = "single-frame-uuid"
            timestamps = np.array([0.0])
            frames = [np.random.rand(1, 32, 32).astype('float32')]
            base_path = os.path.join(temp_dir, "single_frame")
            
            h5_path, yaml_path = save_pca_input_data(UUID, timestamps, frames, base_path)
            
            with h5py.File(h5_path, 'r') as f:
                assert f['frames'].shape == (1, 32, 32)
                assert f['timestamps'].shape == (1,)

    def test_save_pca_input_data_validation_errors(self):
        """Test validation and error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            UUID = "validation-test"
            timestamps = np.linspace(0, 1, 100)
            base_path = os.path.join(temp_dir, "validation_test")
            
            # Test wrong timestamps type
            with pytest.raises(TypeError, match="timestamps must be a numpy array"):
                save_pca_input_data(UUID, [0, 1, 2], [], base_path)
            
            # Test empty timestamps
            with pytest.raises(ValueError, match="timestamps array cannot be empty"):
                save_pca_input_data(UUID, np.array([]), [], base_path)
            
            # Test mismatched frame count
            frames = [np.random.rand(50, 32, 32)]  # 50 frames
            with pytest.raises(ValueError, match="Number of frames .* must match number of timestamps"):
                save_pca_input_data(UUID, timestamps, frames, base_path)  # 100 timestamps
            
            # Test wrong frame dimensions
            bad_frames = [np.random.rand(100, 32)]  # 2D instead of 3D
            with pytest.raises(ValueError, match="Each frame batch must be 3D"):
                save_pca_input_data(UUID, timestamps, bad_frames, base_path)
            
            # Test inconsistent frame dimensions
            inconsistent_frames = [
                np.random.rand(50, 32, 32),
                np.random.rand(50, 64, 64)  # Different size
            ]
            with pytest.raises(ValueError, match="All frames must have same dimensions"):
                save_pca_input_data(UUID, timestamps, inconsistent_frames, base_path)
            
            # Test empty frames iterable
            empty_frames = []
            with pytest.raises(ValueError, match="No frames found in the frames iterable"):
                save_pca_input_data(UUID, np.array([0.0]), empty_frames, base_path)

    def test_save_pca_input_data_generator_input(self):
        """Test with generator as frames input (true streaming)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            UUID = "generator-test"
            timestamps = np.linspace(0, 1, 150)
            base_path = os.path.join(temp_dir, "generator_test")
            
            # Create a generator that yields frame batches
            def frame_generator():
                yield np.random.rand(75, 48, 48).astype('float32')
                yield np.random.rand(75, 48, 48).astype('float32')
            
            h5_path, yaml_path = save_pca_input_data(UUID, timestamps, frame_generator(), base_path)
            
            with h5py.File(h5_path, 'r') as f:
                assert f['frames'].shape == (150, 48, 48)

    def test_save_pca_input_data_list_input(self):
        """Test with list as frames input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            UUID = "list-test"  
            timestamps = np.linspace(0, 0.5, 60)
            base_path = os.path.join(temp_dir, "list_test")
            
            # Use regular Python list
            frames_list = [
                np.ones((20, 40, 40), dtype='float32') * 0.5,
                np.ones((20, 40, 40), dtype='float32') * 1.0, 
                np.ones((20, 40, 40), dtype='float32') * 1.5
            ]
            
            h5_path, yaml_path = save_pca_input_data(UUID, timestamps, frames_list, base_path)
            
            with h5py.File(h5_path, 'r') as f:
                saved_frames = f['frames'][()]
                assert saved_frames.shape == (60, 40, 40)
                
                # Verify the values were preserved
                np.testing.assert_array_equal(saved_frames[:20], 0.5)
                np.testing.assert_array_equal(saved_frames[20:40], 1.0)
                np.testing.assert_array_equal(saved_frames[40:], 1.5)

    def test_save_pca_input_data_different_dtypes(self):
        """Test with different input data types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            UUID = "dtype-test"
            timestamps = np.linspace(0, 1, 50)
            base_path = os.path.join(temp_dir, "dtype_test")
            
            # Test with int input (should be converted to float32)
            frames = [np.random.randint(0, 256, (50, 28, 28), dtype='uint8')]
            
            h5_path, yaml_path = save_pca_input_data(UUID, timestamps, frames, base_path)
            
            with h5py.File(h5_path, 'r') as f:
                saved_frames = f['frames'][()]
                assert saved_frames.dtype == np.float32  # Should be converted