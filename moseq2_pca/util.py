import os
import cv2
import h5py
import time
import dask
import click
import psutil
import warnings
import platform
import subprocess
import numpy as np
import scipy.signal
from copy import deepcopy
import ruamel.yaml as yaml
from tqdm.auto import tqdm
from dask.distributed import Client
from dask_jobqueue import SLURMCluster


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
    '''
    Recursively find h5 files, along with yaml files with the same basename

    Parameters
    ----------
    root_dir (str or os.Pathlike): path to directory to start recursive search
    ext (str): extension to search for, e.g. .h5
    yaml_string (str): a format to use to name yaml files

    Returns
    -------
    h5s (list): list of h5 file paths
    dicts (list): list of dicts containing metadata file contents
    yamls (list): list of yaml file paths
    '''

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
                    else:
                        warnings.warn('Already found uuid {}, file {} is likely a dupe, skipping...'.format(dct['uuid'], os.path.join(root, file)))
            except OSError:
                click.echo('Error loading {}'.format(os.path.join(root, file)))

    return h5s, dicts, yamls


def gauss_smooth(signal, win_length=None, sig=1.5, kernel=None):
    '''
    Perform Gaussian Smoothing on a 1D signal.

    Parameters
    ----------
    signal (1d numpy array): signal to perform smoothing
    win_length (int): window_size for gaussian kernel filter
    sig (float): variance of 1d gaussian kernel.
    kernel (tuple): kernel size to use for smoothing

    Returns
    -------
    result (1d numpy array): smoothed signal
    '''
    if kernel is None:
        kernel = gaussian_kernel1d(n=win_length, sig=sig)

    result = scipy.signal.convolve(signal, kernel, mode='same', method='direct')

    return result


def gaussian_kernel1d(n=None, sig=3):
    '''
    Get 1D gaussian kernel.

    Parameters
    ----------
    n (int): number of points to use.
    sig (int): variance of kernel to use.

    Returns
    -------
    kernel (1d array): 1D numpy kernel.
    '''

    if n is None:
        n = np.ceil(sig * 4)

    points = np.arange(-n, n)

    kernel = np.exp(-(points**2.0) / (2.0 * sig**2.0))
    kernel /= np.sum(kernel)

    return kernel


def clean_frames(frames, medfilter_space=None, gaussfilter_space=None,
                 medfilter_time=None, gaussfilter_time=None, detrend_time=None,
                 tailfilter=None, tail_threshold=5):
    '''
    Filters spatial/temporal noise from frames using Median and Gaussian filters,
    given kernel sizes for each respective requested filter.

    Parameters
    ----------
    frames (3D numpy array): frames to filter.
    medfilter_space (list): median spatial filter kernel.
    gaussfilter_space (list): gaussian spatial filter kernel.
    medfilter_time (list): median temporal filter.
    gaussfilter_time (list): gaussian temporal filter.
    detrend_time (int): number of frames to lag for.
    tailfilter (int): size of tail-filter kernel.
    tail_threshold (int): threshold value to use for tail filtering

    Returns
    -------
    out (3D numpy array): filtered frames.
    '''

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
    '''
    Selects Structuring Element Shape

    Parameters
    ----------
    string (str): e for Ellipse, r for Rectangle
    size (tuple): size of StructuringElement

    Returns
    -------
    strel (cv2.StructuringElement): returned StructuringElement with specified size.
    '''

    if string is None or 'none' in string or np.all(np.array(size) == 0) or len(string) == 0:
        strel = None
    elif string[0].lower() == 'e':
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    elif string[0].lower() == 'r':
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    else:
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    return strel


def insert_nans(timestamps, data, fps=30):
    '''
    Fills NaN values with 0 in timestamps.

    Parameters
    ----------
    timestamps (1D array): timestamp time-strs
    data (1D array): timestamp values
    fps (int): frames per second

    Returns
    -------
    filled_data (1D array): filled missing timestamp values.
    data_idx (1D array): indices of inserted 0s
    filled_timestamps (1D array): filled timestamp-strs
    '''

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
        if idx < len(missing_frames): # ensures ninserts value remains an int
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
    '''
    Reads yaml file and returns dictionary representation of file contents.

    Parameters
    ----------
    yaml_file (str): path to yaml file

    Returns
    -------
    return_dict (dict): dict of yaml file contents
    '''

    try:
        with open(yaml_file, 'r') as f:
            dat = f.read()
            try:
                return_dict = yaml.safe_load(dat)
            except yaml.constructor.ConstructorError:
                return_dict = yaml.safe_load(dat)
    except IOError:
        return_dict = {}

    return return_dict


def get_timestamp_path(h5file):
    '''
    Return path within h5 file that contains the kinect timestamps

    Parameters
    ----------
    h5file (str): path to h5 file.

    Returns
    -------
    (str): path to metadata timestamps within h5 file
    '''

    with h5py.File(h5file, 'r') as f:
        if '/timestamps' in f:
            return '/timestamps'
        elif '/metadata/timestamps' in f:
            return '/metadata/timestamps'
        else:
            click.echo('timestamp key not found!')


def get_metadata_path(h5file):
    '''
    Return path within h5 file that contains the kinect extraction metadata.

    Parameters
    ----------
    h5file (str): path to h5 file.

    Returns
    -------
    (str): path to acquistion metadata within h5 file.
    '''

    with h5py.File(h5file, 'r') as f:
        if '/metadata/acquisition' in f:
            return '/metadata/acquisition'
        elif '/metadata/extraction' in f:
            return '/metadata/extraction'
        else:
            raise KeyError('acquisition metadata not found')


def h5_to_dict(h5file, path):
    '''
    Reads all contents from h5 and returns them in a nested dict object.

    Parameters
    ----------
    h5file (str): path to h5 file
    path (str): path to group within h5 file

    Returns
    -------
    ans (dict): dictionary of all h5 group contents
    '''

    ans = {}

    if type(h5file) is str:
        with h5py.File(h5file, 'r') as f:
            ans = h5_to_dict(f, path)
            return ans

    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[...]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = h5_to_dict(h5file, path + key + '/')
    return ans


def set_dask_config(memory={'target': 0.85, 'spill': False, 'pause': False, 'terminate': 0.95}):
    '''
    Helper function to set the dask memory constraints.
    Ensures there are no memory overflows.

    Parameters
    ----------
    memory (dict): default dictionary containing memory usage parameters to initialize dask with.

    Returns
    -------
    None
    '''
    
    memory = {f'distributed.worker.memory.{k}': v for k, v in memory.items()}
    dask.config.set(memory)
    dask.config.set({'optimization.fuse.ave-width': 5})


def get_env_cpu_and_mem():
    '''
    Gets the amount of available memory and CPUs for dask to use.
    Checks if command is launched in a slurm environment,
    and returns the memory and number of CPUs available accordingly.

    Returns
    -------
    mem (int): Amount of virtual memory (in bytes) available to dask.
    cpu (int): Number of CPUs to initialize dask using.

    '''

    is_slurm = os.environ.get('SLURM_JOBID', False)

    if is_slurm:
        click.echo('Detected slurm environment, using "sacct" to detect cpu and memory requirements')
        cmd = f'sacct -j {is_slurm} --format AllocCPUS,ReqMem -X -n -p'
        output = subprocess.check_output(cmd.split(' '))
        output = output.decode('utf-8').strip().split('|')
        cpu, mem, _ = output
        cpu = int(cpu)
        if 'G' in mem:
            mem = float(mem[:mem.index('G')]) * 1e9
        elif 'M' in mem:
            mem = float(mem[:mem.index('M')]) * 1e6
    else:
        mem = psutil.virtual_memory().available * 0.8
        cpu = max(1, psutil.cpu_count() - 1)

    return mem, cpu


def initialize_dask(nworkers=50, processes=1, memory='4GB', cores=1,
                    wall_time='01:00:00', queue='debug', local_processes=False,
                    cluster_type='local', timeout=10,
                    cache_path=os.path.expanduser('~/moseq2_pca'),
                    dashboard_port='8787', data_size=None, **kwargs):
    '''
    Initialize dask client, cluster, workers, etc.

    Parameters
    ----------
    nworkers (int): number of dask workers to initialize
    processes (int): number of processes per worker
    memory (str): amount of memory to allocate to dask cluster
    cores (int): number of cores to use.
    wall_time (str): amount of time to allow program to run
    queue (str): logging mode
    local_processes (bool): flag to use processes or threads when using a local cluster
    cluster_type (str): indicate what cluster to use (local or slurm)
    scheduler (str): indicate what scheduler to use
    timeout (int): how many minutes to wait for workers to initialize
    cache_path (str or Pathlike): path to store cached data
    dashboard_port (str): port number to find dask statistics
    data_size (float): size of the dask array in number of bytes.
    kwargs: extra keyward arguments

    Returns
    -------
    client (dask Client): initialized Client
    cluster (dask Cluster): initialized Cluster
    workers (dask Workers): intialized workers
    '''

    click.echo(f'Access dask dashboard at localhost:{dashboard_port}')

    if cluster_type == 'local':
        warnings.simplefilter('ignore')

        max_mem, max_cpu = get_env_cpu_and_mem()
        overhead = 0.8e9  # memory overhead for each worker; approximate

        # if we don't know the size of the dataset, fall back onto this
        if data_size is None:
            optimal_workers = (max_mem // overhead) - 1
        else:
            click.echo(f'Using dataset size ({round(data_size / 1e9, 2)}GB) to set optimal parameters')
            # set optimal workers to handle incoming data
            optimal_workers = ((max_mem - data_size) // overhead) - 1

        optimal_workers = max(1, optimal_workers)

        # set number of workers to optimal workers, or total number of CPUs
        # if there are fewer CPUs present than optimal workers
        nworkers = int(min(max(1, max_cpu - 1), optimal_workers))

        # display some diagnostic info
        click.echo(f'Setting number of workers to: {nworkers}')
        click.echo(f'Overriding memory per worker to {round(max_mem / 1e9, 2)}GB')

        client = Client(processes=local_processes,
                        threads_per_worker=1,
                        memory_limit=max_mem,
                        n_workers=nworkers,
                        dashboard_address=dashboard_port,
                        local_directory=cache_path,
                        **kwargs)
        cluster = client.cluster

    elif cluster_type == 'slurm':

        cluster = SLURMCluster(processes=processes,
                               n_workers=nworkers,
                               cores=cores,
                               memory=memory,
                               queue=queue,
                               walltime=wall_time,
                               local_directory=cache_path,
                               dashboard_address=dashboard_port,
                               **kwargs)
        client = Client(cluster)
    else:
        raise NotImplementedError('Specified cluster not supported. Supported types are: "slurm", "local"')

    if client is not None:

        client_info = client.scheduler_info()
        if 'services' in client_info.keys() and 'bokeh' in client_info['services'].keys():
            ip = client_info['address'].split('://')[1].split(':')[0]
            port = client_info['services']['bokeh']
            hostname = platform.node()
            click.echo(f'Web UI served at {ip}:{port} (if port forwarding use internal IP not localhost)')
            click.echo(f'Tunnel command:\n ssh -NL {port}:{ip}:{port} {hostname}')
            click.echo(f'Tunnel command (gcloud):\n gcloud compute ssh {hostname} -- -NL {port}:{ip}:{port}')

    if cluster_type == 'slurm':

        active_workers = len(client.scheduler_info()['workers'])
        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pbar = tqdm(total=nworkers, desc="Intializing workers")

            elapsed_time = (time.time() - start_time) / 60

            while active_workers < nworkers and elapsed_time < timeout:
                tmp = len(client.scheduler_info()['workers'])
                if tmp - active_workers > 0:
                    pbar.update(tmp - active_workers)
                active_workers = tmp
                time.sleep(1)
                elapsed_time = (time.time() - start_time) / 60

            pbar.close()

    workers = cluster.workers

    return client, cluster, workers

def get_rps(frames, rps=600, normalize=True):
    '''
    Get random projections of frames.

    Parameters
    ----------
    frames (2D or 3D numpy array): Frames to get dimensions from.
    rps (int): Number of random projections.
    normalize (bool): indicates whether to normalize frames.

    Returns
    -------
    rproj (2D or 3D numpy array): Computed random projections with same shape as frames
    '''

    if frames.ndim == 3:
        use_frames = frames.reshape(-1, np.prod(frames.shape[1:]))
    elif frames.ndim == 2:
        use_frames = frames

    rproj = use_frames.dot(np.random.randn(use_frames.shape[1], rps).astype('float32'))

    if normalize:
        rproj = scipy.stats.zscore(scipy.stats.zscore(rproj).T)

    return rproj


def get_changepoints(scores, k=5, sigma=3, peak_height=.5, peak_neighbors=1, baseline=True, timestamps=None):
    '''
    Compute changepoints distribution and CP Curve.

    Parameters
    ----------
    scores (3D numpy array): nframes * r * c
    k (int): klags - Lag to use for derivative calculation.
    sigma (int): Standard deviation of gaussian smoothing filter.
    peak_height (float): user-defined peak Changepoint length.
    peak_neighbors (int): number of peaks in the CP curve.
    baseline (bool): normalize data.
    timestamps (array): loaded timestamps.

    Returns
    -------
    cps (2D numpy array): array of values for CP curve
    normed_df (1D numpy array): array of values for bar plot
    '''

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