from moseq2_pca.util import (clean_frames, insert_nans,
                             read_yaml, get_changepoints, get_rps)
from dask.distributed import as_completed, wait, progress
import dask.array.linalg as lng
import dask.array as da
import dask
from dask.diagnostics import ProgressBar
import numpy as np
import h5py
import tqdm
import warnings
import logging


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


def train_pca_dask(dask_array, clean_params, use_fft, rank,
                   cluster_type, client, workers,
                   cache, mask=None, iters=10, recon_pcs=10,
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
    cache (str): path to cache directory
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
    rechunked = False
    _, r, c = dask_array.shape
    nfeatures = r * c

    original_chunks = dask_array.chunks[0][0]

    if original_chunks > 100:
        dask_array.rechunk(100, -1, -1)
        rechunked = True

    smallest_chunk = np.min(dask_array.chunks[0])

    if mask is not None:
        missing_data = True
        dask_array[mask] = 0
        mask = mask.reshape(-1, nfeatures)

    if clean_params['gaussfilter_time'] > 0 or np.any(np.array(clean_params['medfilter_time']) > 0):
        dask_array = dask_array.map_overlap(
            clean_frames, depth=(np.minimum(smallest_chunk, 20), 0, 0), boundary='reflect',
            dtype='float32', **clean_params)
    else:
        dask_array = dask_array.map_blocks(clean_frames, dtype='float32', **clean_params)
        # dask_array = clean_frames(dask_array, **clean_params)

    if use_fft:
        print('Using FFT...')
        dask_array = dask_array.map_blocks(
            lambda x: np.fft.fftshift(np.abs(np.fft.fft2(x)), axes=(1, 2)),
            dtype='float32')

    if rechunked:
        dask_array.rechunk(original_chunks, -1, -1)

    # todo, abstract this into another function, add support for missing data
    # (should be simple, just need a mask array, then repeat calculation to convergence)

    dask_array = dask_array.reshape(-1, nfeatures).astype('float32')
    nsamples, nfeatures = dask_array.shape

    if cluster_type == 'slurm':
        print('Cleaning frames...')
        dask_array = client.persist(dask_array)
        if mask is not None:
            mask = client.persist(mask)

    progress(dask_array)
    wait(dask_array)
    mean = dask_array.mean(axis=0)

    if cluster_type == 'slurm':
        mean = client.persist(mean)

    # todo compute reconstruction error

    print('\nComputing SVD...')

    if not missing_data:
        u, s, v = lng.svd_compressed(dask_array-mean, rank, 0)
    else:
        for iter in range(iters):
            u, s, v = lng.svd_compressed(dask_array-mean, rank, 0)
            if iter < iters - 1:
                recon = u[:, :recon_pcs].dot(da.diag(s[:recon_pcs]).dot(v[:recon_pcs, :])) + mean
                recon[recon < min_height] = 0
                recon[recon > max_height] = 0
                dask_array = da.map_blocks(mask_data, dask_array, mask, recon, dtype=dask_array.dtype)
                mean = dask_array.mean(axis=0)

    total_var = dask_array.var(ddof=1, axis=0).sum()

    # if cluster_type == 'local':
    #     with ProgressBar():
    #         s, v, mean, total_var = dask.compute(s, v, mean, total_var,
    #                                              cache=cache)
    # elif cluster_type == 'slurm':

    futures = client.compute([s, v, mean, total_var])

    with ProgressBar():
        progress(futures)
        s, v, mean, total_var = client.gather(futures)

    # cluster.stop_workers(workers)

    print('\nCalculation complete...')

    # correct the sign of the singular vectors

    tmp = np.argmax(np.abs(v), axis=1)
    correction = np.sign(v[np.arange(v.shape[0]), tmp])
    v *= correction[:, None]

    explained_variance = s ** 2 / (nsamples-1)
    explained_variance_ratio = explained_variance / total_var

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
                    save_file, chunk_size, mask_params, missing_data, fps=30):
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
    Returns
    -------
    None
    '''

    with h5py.File('{}.h5'.format(save_file), 'w') as f_scores:
        for h5, yml in tqdm.tqdm(zip(h5s, yamls), total=len(h5s),
                                 desc='Computing scores'):

            data = read_yaml(yml)
            uuid = data['uuid']

            with h5py.File(h5, 'r') as f:

                frames = f['/frames'][...].astype('float32')

                if missing_data:
                    mask = f['/frames_mask'][...]
                    mask = np.logical_and(mask < mask_params['mask_threshold'],
                                          frames > mask_params['mask_height_threshold'])
                    frames[mask] = 0
                    mask = mask.reshape(-1, frames.shape[1] * frames.shape[2])

                frames = clean_frames(frames, **clean_params)

                if use_fft:
                    frames = np.fft.fftshift(np.abs(np.fft.fft2(frames)), axes=(1, 2))

                frames = frames.reshape(-1, frames.shape[1] * frames.shape[2])

                if '/timestamps' in f:
                    # h5 format post v0.1.3
                    timestamps = f['/timestamps'][...] / 1000.0
                elif '/metadata/timestamps' in f:
                    # h5 format pre v0.1.3
                    timestamps = f['/metadata/timestamps'][...] / 1000.0
                else:
                    timestamps = np.arange(frames.shape[0]) / fps

                if '/metadata/acquisition' in f:
                    # h5 format post v0.1.3
                    metadata_name = 'metadata/{}'.format(uuid)
                    f.copy('/metadata/acquisition', f_scores, name=metadata_name)
                elif '/metadata/extraction' in f:
                    # h5 format pre v0.1.3
                    metadata_name = 'metadata/{}'.format(uuid)
                    f.copy('/metadata/extraction', f_scores, name=metadata_name)

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
                   client, fps=30, gui=False):
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
    Returns
    -------
    None
    '''

    logger = logging.getLogger("distributed.utils_perf")
    logger.setLevel(logging.WARNING)

    futures = []
    uuids = []

    for h5, yml in zip(h5s, yamls):
        data = read_yaml(yml)
        uuid = data['uuid']

        dset = h5py.File(h5, mode='r')['/frames']
        frames = da.from_array(dset, chunks=(chunk_size, -1, -1)).astype('float32')

        if missing_data:
            mask_dset = h5py.File(h5, mode='r')['/frames_mask']
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
        total_batches = len(range(0, len(futures), batch_size))

        for i in range(0, len(futures), batch_size):

            futures_batch = client.compute(futures[i:i+batch_size])
            uuids_batch = uuids[i:i+batch_size]
            h5s_batch = h5s[i:i+batch_size]
            keys = [tmp.key for tmp in futures_batch]
            batch_count += 1


            for future, result in as_completed(futures_batch, with_results=True):

                file_idx = keys.index(future.key)

                with h5py.File(h5s_batch[file_idx], mode='r') as f:
                    if '/timestamps' in f:
                        # h5 format post v0.1.3
                        timestamps = f['/timestamps'][...] / 1000.0
                    elif '/metadata/timestamps' in f:
                        # h5 format pre v0.1.3
                        timestamps = f['/metadata/timestamps'][...] / 1000.0
                    else:
                        timestamps = np.arange(frames.shape[0]) / fps

                    if '/metadata/acquisition' in f:
                        # h5 format post v0.1.3
                        metadata_name = 'metadata/{}'.format(uuids_batch[file_idx])
                        f.copy('/metadata/acquisition', f_scores, name=metadata_name)
                    elif '/metadata/extraction' in f:
                        # h5 format pre v0.1.3
                        metadata_name = 'metadata/{}'.format(uuids_batch[file_idx])
                        f.copy('/metadata/extraction', f_scores, name=metadata_name)

                scores, score_idx, _ = insert_nans(data=result, timestamps=timestamps,
                                                   fps=np.round(1 / np.mean(np.diff(timestamps))).astype('int'))

                f_scores.create_dataset('scores/{}'.format(uuids_batch[file_idx]), data=scores,
                                        dtype='float32', compression='gzip')
                f_scores.create_dataset('scores_idx/{}'.format(uuids_batch[file_idx]), data=score_idx,
                                        dtype='float32', compression='gzip')


def get_changepoints_dask(changepoint_params, pca_components, h5s, yamls,
                          save_file, chunk_size, mask_params, missing_data,
                          client, fps=30, pca_scores=None, progress_bar=False, gui=False):
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
    gui (bool): indicate GUI use
    Returns
    -------
    None
    '''

    futures = []
    uuids = []
    nrps = changepoint_params.pop('rps')
    logger = logging.getLogger("distributed.utils_perf")
    logger.setLevel(logging.WARNING)

    for h5, yml in tqdm.tqdm(zip(h5s, yamls), disable=progress_bar, desc='Setting up calculation', total=len(h5s)):
        data = read_yaml(yml)
        uuid = data['uuid']

        with h5py.File(h5, 'r') as f:

            dset = h5py.File(h5, mode='r')['/frames']
            frames = da.from_array(dset, chunks=(chunk_size, -1, -1)).astype('float32')

            if '/timestamps' in f:
                # h5 format post v0.1.3
                timestamps = f['/timestamps'][...] / 1000.0
            elif '/metadata/timestamps' in f:
                # h5 format pre v0.1.3
                timestamps = f['/metadata/timestamps'][...] / 1000.0
            else:
                timestamps = np.arange(frames.shape[0]) / fps

        if missing_data and pca_scores is None:
            raise RuntimeError("Need to compute PC scores to impute missing data")
        elif missing_data:
            mask_dset = h5py.File(h5, mode='r')['/frames_mask']
            mask = da.from_array(mask_dset, chunks=frames.chunks)
            mask = da.logical_and(mask < mask_params['mask_threshold'],
                                  frames > mask_params['mask_height_threshold'])
            frames[mask] = 0
            mask = mask.reshape(-1, frames.shape[1] * frames.shape[2])

            with h5py.File(pca_scores, 'r') as f:
                scores = f['scores/{}'.format(uuid)]
                scores_idx = f['scores_idx/{}'.format(uuid)][...]
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
        total_batches = len(range(0, len(futures), batch_size))

        for i in range(0, len(futures), batch_size):

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