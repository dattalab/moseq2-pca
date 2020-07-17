import dask
import h5py
import logging
import warnings
import numpy as np
import dask.array as da
from tqdm.auto import tqdm
import dask.array.linalg as lng
from dask.distributed import as_completed, progress
from moseq2_pca.util import (clean_frames, insert_nans, read_yaml, get_changepoints, get_rps)


def mask_data(original_data, mask, new_data):
    '''
    Create a mask subregion given a boolean mask if missing data flag is used.

    Parameters
    ----------
    original_data (3d np.ndarray): input frames
    mask (3d boolean np.ndarray): mask array
    new_data (3d np.ndarray): frames to use

    Returns
    -------
    output (3d np.ndarray): masked data array
    '''

    # need to make a copy otherwise distributed scheduler barfs
    output = original_data.copy()
    output[mask] = new_data[mask]

    return output

def compute_svd(dask_array, mean, rank, iters, missing_data, mask, recon_pcs, min_height, max_height, client):
    '''
    Runs Singular Vector Decomposition on the inputted frames of shape (nframes, nfeatures).
    Data is centered by subtracting it by the mean value of the data. If missing_data == True,
    It will iteratively recompute the svd on the mean-centered data to reconstruct the PCs from
    the missing data until it converges.

    Parameters
    ----------
    dask_array (dask 2d-array): Reshaped input data array of shape (nframes x nfeatures)
    mean (1d array): Means of each row in dask_array.
    rank (int): Rank of the desired thin SVD decomposition.
    iters (int): Number of SVD iterations
    missing_data (bool): Indicates whether to compute SVD with a masked array
    mask (dask 2d-array): None if missing_data == False, else mask array of shape dask_array
    recon_pcs (int): Number of PCs to reconstruct for missing data.
    min_height (int): Minimum height of mouse above the ground, used to filter reconstructed PCs.
    max_height (int): Maximum height of mouse above the ground, used to filter reconstructed PCs.
    client (dask Client): Dask client to process batches.

    Returns
    -------
    s (1d array): computed singular values (eigen-values).
    v (2d array): computed principal components (eigen-vectors).
    mean (1d array): updated mean of dask array if missing_data == True.
    total_var (float): total variance captured by principal components.
    '''

    if not missing_data:
        _, s, v = lng.svd_compressed(dask_array - mean, rank, 0, compute=True)
    else:
        for iter in range(iters):
            u, s, v = lng.svd_compressed(dask_array - mean, rank, 0, compute=True)
            if iter < iters - 1:
                recon = u[:, :recon_pcs].dot(da.diag(s[:recon_pcs]).dot(v[:recon_pcs, :])) + mean
                recon[recon < min_height] = 0
                recon[recon > max_height] = 0
                dask_array = da.map_blocks(mask_data, dask_array, mask, recon, dtype=dask_array.dtype)
                mean = dask_array.mean(axis=0)

    total_var = dask_array.var(ddof=1, axis=0).sum()
    futures = client.compute([s, v, mean, total_var])

    # set notebook=False because progress bar doesn't show up otherwise
    progress(futures, notebook=False)

    s, v, mean, total_var = client.gather(futures)
    return s, v, mean, total_var

def compute_explained_variance(s, nsamples, total_var):
    '''
    Computes the explained variance and explained variance ratio contributed
    by each computed Principal Component.

    Parameters
    ----------
    s (1d array): computed singular values.
    nsamples (int): number of included samples.
    total_var (float): total variance captured by principal components.

    Returns
    -------
    explained_variance (1d-array): list of floats denoting the explained variance per PC.
    explained_variance_ratio (1d-array): list of floats denoting the explained variance ratios per PC.
    '''

    explained_variance = s ** 2 / (nsamples - 1)
    explained_variance_ratio = explained_variance / total_var

    return explained_variance, explained_variance_ratio

def get_timestamps(f, frames, fps=30):
    '''
    Reads the timestamps from a given h5 file.

    Parameters
    ----------
    f (read-open h5py File): open "results_00.h5" h5py.File object in read-mode
    frames (3d-array): list of 2d frames contained in opened h5 File.
    fps (int): frames per second.

    Returns
    -------
    timestamps (1d array): array of timestamps for inputted frames variable
    '''

    if '/timestamps' in f:
        # h5 format post v0.1.3
        timestamps = f['/timestamps'][()] / 1000.0
    elif '/metadata/timestamps' in f:
        # h5 format pre v0.1.3
        timestamps = f['/metadata/timestamps'][()] / 1000.0
    else:
        print('WARNING: timestamps were not found. Using default frame series-ordering.')
        timestamps = np.arange(frames.shape[0]) / fps

    return timestamps

def copy_metadatas_to_scores(f, f_scores, uuid):
    '''
    Copies metadata from individual session extract h5 files to the PCA scores h5 file.

    Parameters
    ----------
    f (read-open h5py File): open "results_00.h5" h5py.File object in read-mode
    f_scores (read-open h5py File): open "pca_scores.h5" h5py.File object in read-mode
    uuid (str): uuid of inputted session h5 "f".

    Returns
    -------
    None
    '''

    if '/metadata/acquisition' in f:
        # h5 format post v0.1.3
        metadata_name = 'metadata/{}'.format(uuid)
        f.copy('/metadata/acquisition', f_scores, name=metadata_name)
    elif '/metadata/extraction' in f:
        # h5 format pre v0.1.3
        metadata_name = 'metadata/{}'.format(uuid)
        f.copy('/metadata/extraction', f_scores, name=metadata_name)

def train_pca_dask(dask_array, clean_params, use_fft, rank,
                   cluster_type, client, workers,
                   mask=None, iters=10, recon_pcs=10,
                   min_height=10, max_height=100):
    '''
    Train PCA using dask arrays.

    Parameters
    ----------
    dask_array (dask array): chunked frames to train PCA
    clean_params (dict): dictionary containing filtering parameters
    use_fft (bool): indicates whether to use 2d-FFT on images.
    rank (int): Matrix rank to use
    cluster_type (str): indicates which cluster to use.
    client (Dask.Client): client object to execute dask operations
    workers (int): number of dask workers
    mask (dask array): dask array of masked data if missing_data parameter==True
    iters (int): number of SVD iterations
    recon_pcs (int): number of PCs to reconstruct. (if missing_data = True)
    min_height (int): minimum mouse height from floor in (mm)
    max_height (int): maximum mouse height from floor in (mm)

    Returns
    -------
    output_dict (dict): dictionary containing PCA training results.
    '''

    logger = logging.getLogger("distributed.utils_perf")
    logger.setLevel(logging.WARNING)

    missing_data = False

    smallest_chunk = np.min(dask_array.chunks[0])

    if mask is not None:
        missing_data = True
        dask_array[mask] = 0
        mask = mask.reshape(len(mask), -1)

    if clean_params['gaussfilter_time'] > 0 or np.any(np.array(clean_params['medfilter_time']) > 0):
        dask_array = dask_array.map_overlap(
            clean_frames, depth=(np.minimum(smallest_chunk, 20), 0, 0), boundary='reflect',
            dtype='float32', **clean_params)
    else:
        dask_array = dask_array.map_blocks(clean_frames, dtype='float32', **clean_params)

    if use_fft:
        print('Using FFT...')
        dask_array = dask_array.map_blocks(
            lambda x: np.fft.fftshift(np.abs(np.fft.fft2(x)), axes=(1, 2)),
            dtype='float32')

    dask_array = dask_array.reshape(len(dask_array), -1).astype('float32')

    if cluster_type == 'slurm':
        print('Cleaning frames...')
        dask_array = client.persist(dask_array)
        if mask is not None:
            mask = client.persist(mask)

    mean = dask_array.mean(axis=0)

    if cluster_type == 'slurm':
        mean = client.persist(mean)

    # todo compute reconstruction error

    print('\nComputing SVD...')

    svd_training_parameters = {
        'dask_array': dask_array,
        'mask': mask,
        'mean': mean,
        'rank': rank,
        'iters': iters,
        'missing_data': missing_data,
        'recon_pcs': recon_pcs,
        'min_height': min_height,
        'max_height': max_height
    }

    s, v, mean, total_var = compute_svd(**svd_training_parameters, client=client)

    print('\nCalculation complete...')

    # correct the sign of the singular vectors
    tmp = np.argmax(np.abs(v), axis=1)
    correction = np.sign(v[np.arange(len(v)), tmp])
    v *= correction[:, None]

    explained_variance, explained_variance_ratio = compute_explained_variance(s, len(dask_array), total_var)

    output_dict = {
        'components': v,
        'singular_values': s,
        'explained_variance': explained_variance,
        'explained_variance_ratio': explained_variance_ratio,
        'mean': mean
    }

    return output_dict


# todo: for applying pca, run once to impute missing data, then get scores
def apply_pca_local(pca_components, h5s, yamls, use_fft, clean_params,
                    save_file, chunk_size, mask_params, missing_data, fps=30,
                    h5_path='/frames', h5_mask_path='/frames_mask'):
    '''
    "Apply" trained PCA on input frame data to obtain PCA Scores
    using local cluster/platform.

    Parameters
    ----------
    pca_components (np.array): array of computed Principal Components
    h5s (list): list of h5 files
    yamls (list): list of yaml files
    use_fft (bool): indicate whether to use 2D-FFT
    clean_params (dict): dictionary containing filtering options
    save_file (str): path to pca_scores filename to save
    chunk_size (int): size of chunks to process
    mask_params (dict): dictionary of masking parameters (if missing data)
    missing_data (bool): indicates whether to use mask arrays.
    fps (int): frames per second
    h5_path (str): path to frames within selected h5 file (default: '/frames')
    h5_mask_path (str): path to masked frames within selected h5 file (default: '/frames_mask')

    Returns
    -------
    None
    '''

    with h5py.File('{}.h5'.format(save_file), 'w') as f_scores:
        for h5, yml in tqdm(zip(h5s, yamls), total=len(h5s), desc='Computing scores'):

            data = read_yaml(yml)
            uuid = data['uuid']

            with h5py.File(h5, 'r') as f:

                frames = f[h5_path][()].astype('float32')

                if missing_data:
                    mask = f[h5_mask_path][()]
                    mask = np.logical_and(mask < mask_params['mask_threshold'],
                                          frames > mask_params['mask_height_threshold'])
                    frames[mask] = 0
                    mask = mask.reshape(-1, frames.shape[1] * frames.shape[2])

                frames = clean_frames(frames, **clean_params)

                if use_fft:
                    frames = np.fft.fftshift(np.abs(np.fft.fft2(frames)), axes=(1, 2))

                frames = frames.reshape(-1, frames.shape[1] * frames.shape[2])

                timestamps = get_timestamps(f, frames, fps)
                copy_metadatas_to_scores(f, f_scores, uuid)

            scores = frames.dot(pca_components.T)

            # if we have missing data, simply fill in, repeat the score calculation,
            # then move on
            if missing_data:
                recon = scores.dot(pca_components)
                recon[recon < mask_params['min_height']] = 0
                recon[recon > mask_params['max_height']] = 0
                frames[mask] = recon[mask]
                scores = frames.dot(pca_components.T)

            scores, score_idx, _ = insert_nans(data=scores, timestamps=timestamps,
                                               fps=np.round(1 / np.mean(np.diff(timestamps))).astype('int'))

            f_scores.create_dataset('scores/{}'.format(uuid), data=scores,
                                    dtype='float32', compression='gzip')
            f_scores.create_dataset('scores_idx/{}'.format(uuid), data=score_idx,
                                    dtype='float32', compression='gzip')


def apply_pca_dask(pca_components, h5s, yamls, use_fft, clean_params,
                   save_file, chunk_size, mask_params, missing_data,
                   client, fps=30, h5_path='/frames', h5_mask_path='/frames_mask'):
    '''
    "Apply" trained PCA on input frame data to obtain PCA Scores using
    Distributed Dask cluster.

    Parameters
    ----------
    pca_components (np.array): array of computed Principal Components
    h5s (list): list of h5 files
    yamls (list): list of yaml files
    use_fft (bool): indicate whether to use 2D-FFT
    clean_params (dict): dictionary containing filtering options
    save_file (str): path to pca_scores filename to save
    chunk_size (int): size of chunks to process
    mask_params (dict): dictionary of masking parameters (if missing data)
    missing_data (bool): indicates whether to use mask arrays.
    fps (int): frames per second
    h5_path (str): path to frames within selected h5 file (default: '/frames')
    h5_mask_path (str): path to masked frames within selected h5 file (default: '/frames_mask')

    Returns
    -------
    None
    '''

    logger = logging.getLogger("distributed.utils_perf")
    logger.setLevel(logging.WARNING)

    futures = []
    uuids = []

    for h5, yml in tqdm(zip(h5s, yamls), total=len(h5s), desc='Loading Data'):
        data = read_yaml(yml)
        uuid = data['uuid']

        dset = h5py.File(h5, mode='r')[h5_path]
        frames = da.from_array(dset, chunks=chunk_size).astype('float32')


        if missing_data:
            mask_dset = h5py.File(h5, mode='r')[h5_mask_path]
            mask = da.from_array(mask_dset, chunks=frames.chunks)
            mask = da.logical_and(mask < mask_params['mask_threshold'],
                                  frames > mask_params['mask_height_threshold'])
            frames[mask] = 0
            mask = mask.reshape(-1, frames.shape[1] * frames.shape[2])

        if clean_params['gaussfilter_time'] > 0 or np.any(np.array(clean_params['medfilter_time']) > 0):
            frames = frames.map_overlap(
                clean_frames, depth=(20, 0, 0), boundary='reflect', dtype='float32', **clean_params)
        else:
            frames = frames.map_blocks(clean_frames, dtype='float32', **clean_params)

        if use_fft:
            frames = frames.map_blocks(
                lambda x: np.fft.fftshift(np.abs(np.fft.fft2(x)), axes=(1, 2)),
                dtype='float32')

        frames = frames.reshape(-1, frames.shape[1] * frames.shape[2])
        scores = frames.dot(pca_components.T)

        if missing_data:
            recon = scores.dot(pca_components)
            recon[recon < mask_params['min_height']] = 0
            recon[recon > mask_params['max_height']] = 0
            frames = da.map_blocks(mask_data, frames, mask, recon, dtype=frames.dtype)
            scores = frames.dot(pca_components.T)

        futures.append(scores)
        uuids.append(uuid)

    # pin the batch size to the number of workers (assume each worker has enough RAM for one session)
    batch_size = len(client.scheduler_info()['workers'])

    with h5py.File('{}.h5'.format(save_file), 'w') as f_scores:

        batch_count = 0
        batches = range(0, len(futures), batch_size)
        for i in tqdm(batches, total=len(batches), desc='Computing scores in batches'):

            futures_batch = client.compute(futures[i:i+batch_size])
            uuids_batch = uuids[i:i+batch_size]
            h5s_batch = h5s[i:i+batch_size]
            keys = [tmp.key for tmp in futures_batch]
            batch_count += 1


            for future, result in as_completed(futures_batch, with_results=True):

                file_idx = keys.index(future.key)

                with h5py.File(h5s_batch[file_idx], mode='r') as f:
                    timestamps = get_timestamps(f, frames, fps)
                    copy_metadatas_to_scores(f, f_scores, uuids_batch[file_idx])

                scores, score_idx, _ = insert_nans(data=result, timestamps=timestamps,
                                                   fps=np.round(1 / np.mean(np.diff(timestamps))).astype('int'))

                f_scores.create_dataset('scores/{}'.format(uuids_batch[file_idx]), data=scores,
                                        dtype='float32', compression='gzip')
                f_scores.create_dataset('scores_idx/{}'.format(uuids_batch[file_idx]), data=score_idx,
                                        dtype='float32', compression='gzip')


def get_changepoints_dask(changepoint_params, pca_components, h5s, yamls,
                          save_file, chunk_size, mask_params, missing_data,
                          client, fps=30, pca_scores=None, progress_bar=False,
                          h5_path='/frames', h5_mask_path='/frames_mask'):
    '''
    Computes model-free changepoints using PCs and PC Scores on distributed dask cluster.

    Parameters
    ----------
    changepoint_params (dict): dict of changepoint parameters
    pca_components (np.array): computed principal components
    h5s (list): list of h5 files
    yamls (list): list of yaml files
    save_file (str): path to save changepoint files
    chunk_size (int): size of chunks to process in dask.
    mask_params (dict): dict of missing_data mask parameters.
    missing_data (bool): indicate whether to use mask_params
    client (dask Client): initialized Dask Client object
    fps (int): frames per second
    pca_scores (np.array): computed principal component scores
    progress_bar (bool): display progress bar
    h5_path (str): path to frames within selected h5 file (default: '/frames')
    h5_mask_path (str): path to masked frames within selected h5 file (default: '/frames_mask')

    Returns
    -------
    None
    '''

    futures = []
    uuids = []
    nrps = changepoint_params.pop('rps')
    logger = logging.getLogger("distributed.utils_perf")
    logger.setLevel(logging.WARNING)

    for h5, yml in tqdm(zip(h5s, yamls), disable=progress_bar, desc='Setting up calculation', total=len(h5s)):
        data = read_yaml(yml)
        uuid = data['uuid']

        with h5py.File(h5, 'r') as f:

            dset = h5py.File(h5, mode='r')[h5_path]
            frames = da.from_array(dset, chunks=chunk_size).astype('float32')

            timestamps = get_timestamps(f, frames, fps)

        if missing_data and pca_scores is None:
            raise RuntimeError("Need to compute PC scores to impute missing data")
        elif missing_data:
            mask_dset = h5py.File(h5, mode='r')[h5_mask_path]
            mask = da.from_array(mask_dset, chunks=frames.chunks)
            mask = da.logical_and(mask < mask_params['mask_threshold'],
                                  frames > mask_params['mask_height_threshold'])
            frames[mask] = 0
            mask = mask.reshape(-1, frames.shape[1] * frames.shape[2])

            with h5py.File(pca_scores, 'r') as f:
                scores = f['scores/{}'.format(uuid)]
                scores_idx = f['scores_idx/{}'.format(uuid)]
                scores = scores[~np.isnan(scores_idx), :]

            if np.sum(frames.chunks[0]) != scores.shape[0]:
                warnings.warn('Chunks do not add up to scores shape in file {}'.format(h5))
                continue

            scores = da.from_array(scores, chunks=(frames.chunks[0], scores.shape[1]))

        frames = frames.reshape(-1, frames.shape[1] * frames.shape[2])

        if missing_data:
            recon = scores.dot(pca_components)
            frames = da.map_blocks(mask_data, frames, mask, recon, dtype=frames.dtype)

        rps = dask.delayed(get_rps, pure=False)(frames, rps=nrps, normalize=True)

        cps = dask.delayed(get_changepoints, pure=True)(rps, timestamps=timestamps, **changepoint_params)

        futures.append(cps)
        uuids.append(uuid)

    # pin the batch size to the number of workers (assume each worker has enough RAM for one session)
    batch_size = 1

    with h5py.File('{}.h5'.format(save_file), 'w') as f_cps:

        f_cps.create_dataset('metadata/fps', data=fps, dtype='float32')

        batch_count = 0

        batches = range(0, len(futures), batch_size)
        for i in tqdm(batches, total=len(batches), desc='Computing changepoints in batches'):

            futures_batch = client.compute(futures[i:i+batch_size])
            uuids_batch = uuids[i:i+batch_size]
            keys = [tmp.key for tmp in futures_batch]
            batch_count += 1

            for future, result in as_completed(futures_batch, with_results=True):

                file_idx = keys.index(future.key)
                if result[0] is not None and result[1] is not None:
                    f_cps.create_dataset('cps_score/{}'.format(uuids_batch[file_idx]), data=result[1],
                                         dtype='float32', compression='gzip')
                    f_cps.create_dataset('cps/{}'.format(uuids_batch[file_idx]), data=result[0] / fps,
                                         dtype='float32', compression='gzip')