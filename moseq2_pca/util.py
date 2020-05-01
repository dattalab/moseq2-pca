from dask_jobqueue import SLURMCluster
from dask.distributed import Client, LocalCluster
import dask.array as da
from chest import Chest
from copy import deepcopy
from tornado import gen
import ruamel.yaml as yaml
import os
import cv2
import h5py
import numpy as np
import click
import scipy.signal
import time
import warnings
import tqdm
import pathlib
import psutil
import platform
import re


# from https://stackoverflow.com/questions/46358797/
# python-click-supply-arguments-and-options-from-a-configuration-file
def command_with_config(config_file_param_name):
    class custom_command_class(click.Command):

        def invoke(self, ctx):
            config_file = ctx.params[config_file_param_name]
            param_defaults = {}
            for param in self.params:
                if type(param) is click.core.Option:
                    param_defaults[param.human_readable_name] = param.default

            if config_file is not None:
                config_data = read_yaml(config_file)
                for param, value in ctx.params.items():
                    if param in config_data:
                        if type(value) is tuple and type(config_data[param]) is int:
                            ctx.params[param] = tuple([config_data[param]])
                        elif type(value) is tuple:
                            ctx.params[param] = tuple(config_data[param])
                        else:
                            ctx.params[param] = config_data[param]

                        if param_defaults[param] != value:
                            ctx.params[param] = value

            return super(custom_command_class, self).invoke(ctx)

    return custom_command_class


def recursive_find_h5s(root_dir=os.getcwd(),
                       ext='.h5',
                       yaml_string='{}.yaml'):
    """Recursively find h5 files, along with yaml files with the same basename
    """
    dicts = []
    h5s = []
    yamls = []
    uuids = []
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

    return h5s, dicts, yamls


def gauss_smooth(signal, win_length=None, sig=1.5, kernel=None):

    if kernel is None:
        kernel = gaussian_kernel1d(n=win_length, sig=sig)

    result = scipy.signal.convolve(signal, kernel, mode='same', method='direct')

#    win_length = len(kernel)

#     result[:win_length] = np.nan
#     result[-win_length:] = np.nan

    return result


def gaussian_kernel1d(n=None, sig=3):

    if n is None:
        n = np.ceil(sig * 4)

    points = np.arange(-n, n)

    kernel = np.exp(-(points**2.0) / (2.0 * sig**2.0))
    kernel /= np.sum(kernel)

    return kernel


def clean_frames(frames, medfilter_space=None, gaussfilter_space=None,
                 medfilter_time=None, gaussfilter_time=None, detrend_time=None,
                 tailfilter=None, tail_threshold=5):

    out = np.copy(frames)

    if tailfilter is not None:
        for i in range(frames.shape[0]):
            mask = cv2.morphologyEx(out[i], cv2.MORPH_OPEN, tailfilter) > tail_threshold
            out[i] = out[i] * mask.astype(frames.dtype)

    if medfilter_space is not None and np.all(np.array(medfilter_space) > 0):
        for i in range(frames.shape[0]):
            for medfilt in medfilter_space:
                out[i] = cv2.medianBlur(out[i], medfilt)

    if gaussfilter_space is not None and np.all(np.array(gaussfilter_space) > 0):
        for i in range(frames.shape[0]):
            out[i] = cv2.GaussianBlur(out[i], (21, 21),
                                      gaussfilter_space[0], gaussfilter_space[1])

    if medfilter_time is not None and np.all(np.array(medfilter_time) > 0):
        for idx, i in np.ndenumerate(frames[0]):
            for medfilt in medfilter_time:
                out[:, idx[0], idx[1]] = \
                    scipy.signal.medfilt(out[:, idx[0], idx[1]], medfilt)

    if gaussfilter_time is not None and gaussfilter_time > 0:
        kernel = gaussian_kernel1d(sig=gaussfilter_time)
        for idx, i in np.ndenumerate(frames[0]):
            out[:, idx[0], idx[1]] = \
                np.convolve(out[:, idx[0], idx[1]], kernel, mode='same')

    if detrend_time is not None and detrend_time > 0:
        kernel = gaussian_kernel1d(sig=detrend_time)
        for idx, i in np.ndenumerate(frames[0]):
            out[:, idx[0], idx[1]] = \
                out[:, idx[0], idx[1]] - gauss_smooth(out[:, idx[0], idx[1]], kernel=kernel)

    return out


def select_strel(string='e', size=(10, 10)):
    if string is None or 'none' in string or np.all(np.array(size) == 0):
        strel = None
    elif string[0].lower() == 'e':
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    elif string[0].lower() == 'r':
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    else:
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    return strel


def insert_nans(timestamps, data, fps=30):

    df_timestamps = np.diff(
        np.insert(timestamps, 0, timestamps[0] - 1.0 / fps))
    missing_frames = np.floor(df_timestamps / (1.0 / fps))
    fill_idx = np.where(missing_frames > 1)[0]
    data_idx = np.arange(len(timestamps)).astype('float64')

    filled_data = deepcopy(data)
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
            np.cumsum(np.ones(ninserts,) * 1.0 / fps)
        filled_data = np.insert(filled_data, idx,
                                np.ones((ninserts, nfeatures)) * np.nan, axis=0)
        filled_timestamps = np.insert(
            filled_timestamps, idx, insert_timestamps)

    if isvec:
        filled_data = np.squeeze(filled_data)

    return filled_data, data_idx, filled_timestamps


def read_yaml(yaml_file):

    try:
        with open(yaml_file, 'r') as f:
            dat = f.read()
            try:
                return_dict = yaml.load(dat, Loader=yaml.RoundTripLoader)
            except yaml.constructor.ConstructorError:
                return_dict = yaml.load(dat, Loader=yaml.Loader)
    except IOError:
        return_dict = {}

    return return_dict


def get_timestamp_path(h5file):
    '''Return path within h5 file that contains the kinect timestamps'''
    with h5py.File(h5file, 'r') as f:
        if '/timestamps' in f:
            return '/timestamps'
        elif '/metadata/timestamps' in f:
            return '/metadata/timestamps'
        else:
            raise KeyError('timestamp key not found')


def get_metadata_path(h5file):
    '''Return path within h5 file that contains the kinect extraction metadata'''
    with h5py.File(h5file, 'r') as f:
        if '/metadata/acquisition' in f:
            return '/metadata/acquisition'
        elif '/metadata/extraction' in f:
            return '/metadata/extraction'
        else:
            raise KeyError('acquisition metadata not found')


def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}

    if type(h5file) is str:
        with h5py.File(h5file, 'r') as f:
            ans = recursively_load_dict_contents_from_group(f, path)
            return ans

    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[...]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + '/')
    return ans


def initialize_dask(nworkers=50, processes=1, memory='4GB', cores=1,
                    wall_time='01:00:00', queue='debug', local_processes=False,
                    cluster_type='local', scheduler='distributed', timeout=10,
                    cache_path=os.path.join(pathlib.Path.home(), 'moseq2_pca'),
                    **kwargs):

    # only use distributed if we need it

    client = None
    workers = None
    cache = None
    cluster = None

    if cluster_type == 'local' and scheduler == 'dask':

        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        cache = Chest(path=cache_path)

    elif cluster_type == 'local' and scheduler == 'distributed':

        ncpus = psutil.cpu_count()
        mem = psutil.virtual_memory().total
        mem_per_worker = np.floor(((mem * .8) / nworkers) / 1e9)
        cur_mem = float(re.search(r'\d+', memory).group(0))

        # TODO: make a decision re: threads here (maybe leave as an option?)

        if cores * nworkers > ncpus or cur_mem > mem_per_worker:

            if cores * nworkers > ncpus:
                cores = 1
                nworkers = ncpus

            if cur_mem > mem_per_worker:
                mem_per_worker = np.round(((mem * .8) / nworkers) / 1e9)
                memory = '{}GB'.format(mem_per_worker)

            warning_string = ("ncpus or memory out of range, setting to "
                              "{} cores, {} workers, {} mem per worker "
                              "\n!!!IF YOU ARE RUNNING ON A CLUSTER MAKE "
                              "SURE THESE SETTINGS ARE CORRECT!!!"
                              ).format(cores, nworkers, memory)
            warnings.warn(warning_string)
            # input('Press ENTER to continue...')

        cluster = LocalCluster(n_workers=nworkers,
                               threads_per_worker=cores,
                               processes=local_processes,
                               local_dir=cache_path,
                               memory_limit=memory,
                               **kwargs)
        client = Client(cluster)

    elif cluster_type == 'slurm':

        cluster = SLURMCluster(processes=processes,
                               cores=cores,
                               memory=memory,
                               queue=queue,
                               walltime=wall_time,
                               local_directory=cache_path,
                               **kwargs)

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

    return client, cluster, workers, cache


# graceful shutdown...
# https://github.com/dask/distributed/issues/1703#issuecomment-361291492
@gen.coroutine
def shutdown_dask(scheduler):
    yield scheduler.retire_workers(workers=scheduler.workers, close_workers=True)
    yield scheduler.close()


def get_rps(frames, rps=600, normalize=True):

    if frames.ndim == 3:
        use_frames = frames.reshape(-1, np.prod(frames.shape[1:]))
    elif frames.ndim == 2:
        use_frames = frames

    rproj = use_frames.dot(np.random.randn(use_frames.shape[1], rps).astype('float32'))

    if normalize:
        rproj = scipy.stats.zscore(scipy.stats.zscore(rproj).T)

    return rproj


def get_rps_dask(frames, client=None, rps=600, chunk_size=5000, normalize=True):

    rps = frames.dot(da.random.normal(0, 1,
                                      size=(frames.shape[1], 600),
                                      chunks=(chunk_size, -1)))
    rps = scipy.stats.zscore(scipy.stats.zscore(rps).T)

    if client is not None:
        rps = client.scatter(rps)

    return rps


def get_changepoints(scores, k=5, sigma=3, peak_height=.5, peak_neighbors=1, baseline=True, timestamps=None):

    if type(k) is not int:
        k = int(k)

    if type(peak_neighbors) is not int:
        peak_neighbors = int(peak_neighbors)

    normed_df = deepcopy(scores)
    nanidx = np.isnan(normed_df)
    normed_df[nanidx] = 0

    if sigma is not None and sigma > 0:
        for i in range(scores.shape[0]):
            normed_df[i, :] = gauss_smooth(normed_df[i, :], sigma)

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

    return cps, normed_df
