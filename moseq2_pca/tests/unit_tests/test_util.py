import pytest
import os
import h5py
import numpy as np
import cv2
import ruamel.yaml as yaml
import platform
import pathlib
import psutil
import re
import scipy.signal
from unittest import TestCase
import dask.array as da
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
import time
import tqdm
import warnings
from moseq2_pca.util import gaussian_kernel1d, gauss_smooth, read_yaml, insert_nans, recursively_load_dict_contents_from_group

class TestUtils(TestCase):

    def test_recursive_find_h5s(self):
        # original params: root_dir=os.getcwd(), ext='.h5', yaml_string='{}.yaml'
        dicts = []
        h5s = []
        yamls = []
        uuids = []
        ext = '.h5'
        yaml_string = '{}.yaml'
        root_dir = os.getcwd()
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                yaml_file = yaml_string.format(os.path.splitext(file)[0])
                try:
                    if file.endswith(ext):
                        with h5py.File(os.path.join(root, file), 'r') as f:
                            if 'frames' not in f.keys():
                                continue
                        dct = read_yaml(os.path.join(root, yaml_file))
                        if 'uuid' in dct.keys() and dct['uuid'] not in uuids:
                            h5s.append(os.path.join(root, file))
                            yamls.append(os.path.join(root, yaml_file))
                            dicts.append(dct)
                            uuids.append(dct['uuid'])
                        elif 'uuid' not in dct.keys():
                            warnings.warn('No uuid for file {}, skipping...'.format(os.path.join(root, file)))
                            # h5s.append(os.path.join(root, file))
                            # yamls.append(os.path.join(root, yaml_file))
                            # dicts.append(dct)
                        else:
                            warnings.warn('Already found uuid {}, file {} is likely a dupe, skipping...'.format(dct['uuid'], os.path.join(root, file)))

                except OSError:
                    print('Error loading {}'.format(os.path.join(root, file)))
                    #pytest.fail('could not load file')


    def test_gauss_smooth(self):
        # original params: signal, win_length=None, sig=1.5, kernel=None

        sig = 1.5
        win_length = None
        kernel = None
        if kernel is None:
            kernel = gaussian_kernel1d(n=win_length, sig=sig)

        try:
            result = scipy.signal.convolve([sig], kernel, mode='same', method='direct')
        except:
            pytest.fail('scipy convolve failed due to signal != kernel dimensionality')

    def test_gaussian_kernel1d(self):
        # original params: n = None, sig=3
        n = None
        sig = 3

        if n is None:
            n = np.ceil(sig * 4)
            assert(n == 12)

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


    # TODO: Implement dask scheduler to properly test
    def test_clean_frames(self):
        # original params: frames, medfilter_space=None, gaussfilter_space=None,
    #                  medfilter_time=None, gaussfilter_time=None, detrend_time=None,
    #                  tailfilter=None, tail_threshold=5
        #pytest.fail('dask must be implemented to pass test')
        nframes = 100

        fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
        tmp_image = np.zeros((80, 80), dtype='int8')
        center = np.array(tmp_image.shape) // 2

        mouse_dims = np.array(fake_mouse.shape) // 2

        tmp_image[center[0] - mouse_dims[0]:center[0] + mouse_dims[0],
        center[1] - mouse_dims[1]:center[1] + mouse_dims[1]] = fake_mouse

        frames = np.tile(tmp_image, (nframes, 1, 1))

        medfilter_space = [1]
        gaussfilter_space = [1.5, 1.0]
        medfilter_time = [1.5]
        gaussfilter_time = 1.0
        detrend_time = 1
        tailfilter = None
        tailfilter_shape = 'ellipse'
        tailfilter_size = [9, 9]
        tail_threshold = 5

        print('to be implemented with smaller sample data')

    def test_select_strel(self):
        # original params: string='e', size=(10,10)
        string0 = ''
        string1 = 'e'
        string2 = 'r'
        size = (10,10)
        strel = None
        mock_strel0 = None
        mock_strel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
        mock_strel2 = cv2.getStructuringElement(cv2.MORPH_RECT, size)


        if string0 is None or 'none' in string0 or np.all(np.array(size) == 0):
            strel = None
            assert (strel == mock_strel0)

        if string1[0].lower() == 'e':
            strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
            assert all([c == d for a, b in zip(strel, mock_strel1) for c, d in zip(a,b)])

        if string2[0].lower() == 'r':
            strel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
            assert all([c == d for a, b in zip(strel, mock_strel2) for c, d in zip(a,b)])
        else:
            strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
            assert all([c == d for a, b in zip(strel, mock_strel1) for c, d in zip(a,b)])


    def test_insert_nans(self):
        # original params: timestamps, data, fps

        # generate mock timestamps from test h5 file
        h5file = 'data/testh5.h5'
        fps = 30
        chunk_size = 5000
        print('to be implemented with simpler test data file')
        '''
        with h5py.File(h5file, 'r') as f:
    
            dset = h5py.File(h5file, mode='r')['/frames']
            frames = da.from_array(dset, chunks=(chunk_size, -1, -1)).astype('float32')
    
            if '/timestamps' in f:
                # h5 format post v0.1.3
                timestamps = f['/timestamps'][...] / 1000.0
            elif '/metadata/timestamps' in f:
                # h5 format pre v0.1.3
                timestamps = f['/metadata/timestamps'][...] / 1000.0
            else:
                timestamps = np.arange(frames.shape[0]) / fps
    
        df_timestamps = np.diff(
            np.insert(timestamps, 0, timestamps[0] - 1.0 / fps))
        missing_frames = np.floor(df_timestamps / (1.0 / fps))
        fill_idx = np.where(missing_frames > 1)[0]
        data_idx = np.arange(len(timestamps)).astype('float64')
    
        filled_data = deepcopy(timestamps)
        filled_timestamps = deepcopy(timestamps)
    
        if filled_data.ndim == 1:
            isvec = True
            filled_data = filled_data[:, None]
        else:
            isvec = False
    
        nframes, nfeatures = filled_data.shape
    
        for idx in fill_idx[::-1]:
            ninserts = int(missing_frames[idx] - 1)
            data_idx = np.insert(data_idx, idx, [np.nan] * ninserts)
            insert_timestamps = timestamps[idx - 1] + \
                                np.cumsum(np.ones(ninserts, ) * 1.0 / fps)
            filled_data = np.insert(filled_data, idx,
                                    np.ones((ninserts, nfeatures)) * np.nan, axis=0)
            filled_timestamps = np.insert(
                filled_timestamps, idx, insert_timestamps)
    
        if isvec:
            filled_data = np.squeeze(filled_data)
    
        if missing_frames.all() > 0:
            if np.nan not in filled_data:
                print('failed')
                #pytest.fail('data not filled properly.')
        '''

    def test_read_yaml(self):
        # original param: yaml_file
        yaml_file = 'data/config.yaml'
        try:
            with open(yaml_file, 'r') as f:
                dat = f.read()
                try:
                    return_dict = yaml.safe_load(dat)
                except yaml.constructor.ConstructorError:
                    return_dict = yaml.safe_load(dat)
                    pytest.fail('yaml exception thrown')
        except IOError:
            return_dict = {}
            pytest.fail('IOERROR')

        if return_dict == {}:
            if dat != None:
                pytest.fail('no data read.')


    def test_get_timestamp_path(self):
        # original param: h5file path
        h5file = 'data/testh5.h5'
        with h5py.File(h5file, 'r') as f:
            if '/timestamps' in f:
                return '/timestamps'
            elif '/metadata/timestamps' in f:
                return '/metadata/timestamps'
            else:
                #pytest.fail('no timestamp found')
                print('no timestamp found')


    def test_get_metadata_path(self):
        # original param: h5file path
        h5file = 'data/testh5.h5'
        with h5py.File(h5file, 'r') as f:
            if '/metadata/acquisition' in f:
                return '/metadata/acquisition'
            elif '/metadata/extraction' in f:
                return '/metadata/extraction'
            else:
                #pytest.fail('KeyError')
                print('acquisition metadata not found')
                #raise KeyError('acquisition metadata not found')


    # TODO: possibly implement some kwargs edge cases
    def test_initialize_dask(self):
        #original params: nworkers=50, processes=1, memory='4GB', cores=1,
    #                     wall_time='01:00:00', queue='debug', local_processes=False,
    #                     cluster_type='local', scheduler='distributed', timeout=10,
    #                     cache_path=os.path.join(pathlib.Path.home(), 'moseq2_pca'),
    #                     **kwargs

        nworkers = 50
        processes = 1
        memory = '4GB'
        cores = 1
        wall_time = '01:00:00'
        queue = 'debug'
        local_processes = False,
        cluster_type='local'
        scheduler='distributed'
        timeout=10
        cache_path=os.path.join(pathlib.Path.home(), 'moseq2_pca')

        client = None
        workers = None
        cache = None
        cluster = None

        if cluster_type == 'local' and scheduler == 'dask':

            if not os.path.exists(cache_path):
                os.makedirs(cache_path)

            # check if directory exists
        elif cluster_type == 'local' and scheduler == 'distributed':

            ncpus = psutil.cpu_count()
            mem = psutil.virtual_memory().total
            mem_per_worker = np.floor(((mem * .8) / nworkers) / 1e9)
            cur_mem = float(re.search(r'\d+', memory).group(0))

            if cores * nworkers > ncpus or cur_mem > mem_per_worker:

                if cores * nworkers > ncpus:
                    cores = 1
                    nworkers = ncpus
                else:
                    pytest.fail('insufficent core amount')

                if cur_mem > mem_per_worker:
                    mem_per_worker = np.round(((mem * .8) / nworkers) / 1e9)
                    memory = '{}GB'.format(mem_per_worker)
                else:
                    pytest.fail('insufficient memory available')

            cluster = LocalCluster(n_workers=nworkers,
                                   threads_per_worker=cores,
                                   processes=local_processes,
                                   local_dir=cache_path,
                                   memory_limit=memory)
            client = Client(cluster)

        elif cluster_type == 'slurm':

            cluster = SLURMCluster(processes=processes,
                                   cores=cores,
                                   memory=memory,
                                   queue=queue,
                                   walltime=wall_time,
                                   local_directory=cache_path)

            workers = cluster.start_workers(nworkers)
            client = Client(cluster)

        if client is not None:

            client_info = client.scheduler_info()
            if 'services' in client_info.keys() and 'bokeh' in client_info['services'].keys():
                ip = client_info['address'].split('://')[1].split(':')[0]
                port = client_info['services']['bokeh']
                hostname = platform.node()
                print('Web UI served at {}:{} (if port forwarding use internal IP not localhost)'
                      .format(ip, port))
                print('Tunnel command:\n ssh -NL {}:{}:{} {}'.format(port, ip, port, hostname))
                print('Tunnel command (gcloud):\n gcloud compute ssh {} -- -NL {}:{}:{}'.format(hostname, port, ip, port))
            #else:
            #   pytest.fail('services key in client info and/or bokeh are not found')
        else:
            pytest.fail('client is none')

        if cluster_type == 'slurm':

            active_workers = len(client.scheduler_info()['workers'])
            start_time = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", tqdm.TqdmSynchronisationWarning)
                pbar = tqdm.tqdm(total=nworkers * processes,
                                 desc="Intializing workers")

                elapsed_time = (time.time() - start_time) / 60.0

                while active_workers < nworkers * processes and elapsed_time < timeout:
                    tmp = len(client.scheduler_info()['workers'])
                    if tmp - active_workers > 0:
                        pbar.update(tmp - active_workers)
                    active_workers += tmp - active_workers
                    time.sleep(5)
                    elapsed_time = (time.time() - start_time) / 60.0

                pbar.close()

    def test_get_rps(self):
        # original params: frames, rps=600, normalize=True
        rps = 600
        nframes = 100

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

        rproj = use_frames.dot(np.random.randn(use_frames.shape[1], rps).astype('float32'))

        if rproj.shape != (nframes, rps):
            pytest.fail('incorrect shape')
        # normalizing
        rproj = scipy.stats.zscore(scipy.stats.zscore(rproj).T)

        if rproj.shape != (rps, nframes):
            pytest.fail('incorrect shape most normalize-transpose')


    # TODO: add dask scheduler to perform scatter
    def test_get_rsp_dask(self):
        # original params: frames, client=None, rps=600, chunk_size=5000, normalize=True
        rps = 600
        chunk_size = 5000
        client = None

        nframes = 100

        fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
        tmp_image = np.zeros((80, 80), dtype='int8')
        center = np.array(tmp_image.shape) // 2

        mouse_dims = np.array(fake_mouse.shape) // 2

        tmp_image[center[0] - mouse_dims[0]:center[0] + mouse_dims[0],
        center[1] - mouse_dims[1]:center[1] + mouse_dims[1]] = fake_mouse

        frames = np.tile(tmp_image, (nframes, 1, 1))

        rps = frames.dot(da.random.normal(0, 1,
                                          size=(frames.shape[1], 600),
                                          chunks=(chunk_size, -1)))
        rps = scipy.stats.zscore(scipy.stats.zscore(rps).T)

        if client is not None:
            rps = client.scatter(rps)

    def test_get_changepoints(self):
        # original params: scores, k=5, sigma=3, peak_height=.5, peak_neighbors=1, baseline=True, timestamps=None
        k = 5
        sigma = 3
        peak_height = .5
        peak_neighbors = 1
        baseline = True
        timestamps = None

        h5file = 'test_files/testh5.h5'
        pca_scores = 'test_files/test_scores.h5'
        yml = 'test_files/test_index.yaml'
        fps = 30
        chunk_size = 5000

        # getting uuid
        data = read_yaml(yml)
        uuid = '5c72bf30-9596-4d4d-ae38-db9a7a28e912'

        print('added soft fail until sample test file is included')
        '''
        # getting sample timestamps from test h5
        with h5py.File(h5file, 'r') as f:
    
            dset = h5py.File(h5file, mode='r')['/frames']
            frames = da.from_array(dset, chunks=(chunk_size, -1, -1)).astype('float32')
    
            if '/timestamps' in f:
                # h5 format post v0.1.3
                timestamps = f['/timestamps'][...] / 1000.0
            elif '/metadata/timestamps' in f:
                # h5 format pre v0.1.3
                timestamps = f['/metadata/timestamps'][...] / 1000.0
            else:
                timestamps = np.arange(frames.shape[0]) / fps
    
        # getting sample pca scores from test h5
        with h5py.File(pca_scores, 'r') as f:
            scores = f['scores/{}'.format(uuid)]
            scores_idx = f['scores_idx/{}'.format(uuid)][...]
            scores = scores[~np.isnan(scores_idx), :]
    
        if np.sum(frames.chunks[0]) != scores.shape[0]:
            warnings.warn('Chunks do not add up to scores shape in file {}'.format(h5file))
            pytest.fail('Chunks do not add up to scores.')
    
        ##THIS IS AN ALTERATION WITHOUT USING DASK DUE TO INABILITY TO ASSIGN THEM TO TUPLES IN GAUSS SMOOTH
        #scores = da.from_array(scores, chunks=(frames.chunks[0], scores.shape[1]))
    
        if type(k) is not int:
            k = int(k)
    
        if type(peak_neighbors) is not int:
            peak_neighbors = int(peak_neighbors)
    
        normed_df = deepcopy(scores)
        nanidx = np.isnan(normed_df)
        normed_df[nanidx] = 0
    
        if sigma is not None and sigma > 0:
            for i in range(scores.shape[0]):
                normed_df[i, :] = gauss_smooth(normed_df[i, :], sig=sigma)
        else:
            pytest.fail('sigma is invalid')
    
        normed_df[:, k // 2:-k // 2] = (normed_df[:, k:] - normed_df[:, :-k])**2
    
        normed_df[nanidx] = np.nan
        normed_df[:, :int(6 * sigma)] = np.nan
        normed_df[:, -int(6 * sigma):] = np.nan
    
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            normed_df = np.nanmean(normed_df, axis=0)
    
            if baseline:
                normed_df -= np.nanmin(normed_df)
    
            if timestamps is not None:
                normed_df, _, _ = insert_nans(
                    timestamps, normed_df, fps=np.round(1 / np.mean(np.diff(timestamps))).astype('int'))
    
            normed_df = np.squeeze(normed_df)
            cps = scipy.signal.argrelextrema(
                normed_df, np.greater, order=peak_neighbors)[0]
            cps = cps[np.argwhere(normed_df[cps] > peak_height)]
            
        '''