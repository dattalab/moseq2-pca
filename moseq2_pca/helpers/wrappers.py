import os
import h5py
import click
import logging
import datetime
import warnings
import dask.array as da
import ruamel.yaml as yaml
from moseq2_pca.viz import display_components, scree_plot, changepoint_dist
from moseq2_pca.helpers.data import setup_cp_command, get_pca_yaml_data, load_pcs_for_cp
from moseq2_pca.pca.util import apply_pca_dask, apply_pca_local, train_pca_dask, get_changepoints_dask
from moseq2_pca.util import recursive_find_h5s, select_strel, initialize_dask, set_dask_config, \
            h5_to_dict, get_timestamp_path, get_metadata_path

def train_pca_wrapper(input_dir, config_data, output_dir, output_file):
    '''
    Wrapper function to train PCA.

    Parameters
    ----------
    input_dir (int): path to directory containing all h5+yaml files
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    output_dir (str): path to directory to store PCA data
    output_file (str): pca model filename

    Returns
    -------
    config_data (dict): updated config_data variable to write back in GUI API
    '''

    set_dask_config()

    if config_data['missing_data'] and config_data['use_fft']:
        raise NotImplementedError("FFT and missing data not implemented yet")

    params = config_data
    # find directories with .dat files that either have incomplete or no extractions
    h5s, dicts, yamls = recursive_find_h5s(input_dir)
    timestamp = f'{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}'

    params['start_time'] = timestamp
    params['inputs'] = h5s

    output_dir = os.path.abspath(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_file = os.path.join(output_dir, output_file)

    if os.path.exists(f'{save_file}.h5'):
        click.echo(f'The file {save_file}.h5 already exists.\nWould you like to overwrite it? [y -> yes, else -> exit]\n')
        ow = input()
        if ow.lower() != 'y':
            return config_data

    config_store = '{}.yaml'.format(save_file)
    with open(config_store, 'w') as f:
        yaml.safe_dump(params, f)

    tailfilter = select_strel((config_data['tailfilter_shape'], config_data['tailfilter_size']))

    clean_params = {
        'gaussfilter_space': config_data['gaussfilter_space'],
        'gaussfilter_time': config_data['gaussfilter_time'],
        'tailfilter': tailfilter,
        'medfilter_time': config_data['medfilter_time'],
        'medfilter_space': config_data['medfilter_space']
    }

    logging.basicConfig(filename=f'{output_dir}/train.log', level=logging.ERROR)

    dsets = [h5py.File(h5, mode='r')[config_data['h5_path']] for h5 in h5s]
    arrays = [da.from_array(dset, chunks=config_data['chunk_size']) for dset in dsets]
    stacked_array = da.concatenate(arrays, axis=0)
    stacked_array[stacked_array < config_data['min_height']] = 0
    stacked_array[stacked_array > config_data['max_height']] = 0

    config_data['data_size'] = stacked_array.nbytes

    client, cluster, workers = \
        initialize_dask(cluster_type=config_data['cluster_type'],
                        nworkers=config_data['nworkers'],
                        cores=config_data['cores'],
                        processes=config_data['processes'],
                        memory=config_data['memory'],
                        wall_time=config_data['wall_time'],
                        queue=config_data['queue'],
                        timeout=config_data['timeout'],
                        cache_path=config_data['dask_cache_path'],
                        local_processes=config_data['local_processes'],
                        dashboard_port=config_data['dask_port'],
                        data_size=config_data['data_size'])

    click.echo(f'Processing {len(stacked_array)} total frames')

    if config_data['missing_data']:
        mask_dsets = [h5py.File(h5, mode='r')[config_data['h5_mask_path']] for h5 in h5s]
        mask_arrays = [da.from_array(dset, chunks=config_data['chunk_size']) for dset in mask_dsets]
        stacked_array_mask = da.concatenate(mask_arrays, axis=0).astype('float32')
        stacked_array_mask = da.logical_and(stacked_array_mask < config_data['mask_threshold'],
                                            stacked_array > config_data['mask_height_threshold'])
        click.echo('Loaded mask for missing data')

    else:
        stacked_array_mask = None

    try:
        output_dict = \
            train_pca_dask(dask_array=stacked_array, mask=stacked_array_mask,
                           clean_params=clean_params, use_fft=config_data['use_fft'],
                           rank=config_data['rank'], cluster_type=config_data['cluster_type'],
                           min_height=config_data['min_height'],
                           max_height=config_data['max_height'], client=client,
                           iters=config_data['missing_data_iters'], workers=workers,
                           recon_pcs=config_data['recon_pcs'])
    except Exception as e:
        logging.error(e)
        logging.error(e.__traceback__)
        click.echo('Training interrupted. Closing Dask Client. You may find logs of the error here:')
        click.echo('---- ', os.path.join(output_dir, 'train.log'))
        client.close(timeout=config_data['timeout'])
        cluster.close(timeout=config_data['timeout'])

    try:
        plt, _ = display_components(output_dict['components'], headless=True)
        plt.savefig(f'{save_file}_components.png')
        plt.savefig(f'{save_file}_components.pdf')
        plt.close()
    except Exception as e:
        logging.error(e)
        logging.error(e.__traceback__)
        click.echo('could not plot components')
        click.echo('You may find error logs here:', os.path.join(output_dir, 'train.log'))
    try:
        plt = scree_plot(output_dict['explained_variance_ratio'], headless=True)
        plt.savefig(f'{save_file}_scree.png')
        plt.savefig(f'{save_file}_scree.pdf')
        plt.close()
    except Exception as e:
        logging.error(e)
        logging.error(e.__traceback__)
        click.echo('could not plot scree')
        click.echo('You may find error logs here:', os.path.join(output_dir, 'train.log'))

    with h5py.File(f'{save_file}.h5', 'w') as f:
        for k, v in output_dict.items():
            f.create_dataset(k, data=v, compression='gzip', dtype='float32')

    config_data['pca_file'] = f'{save_file}.h5'

    if client is not None:
        try:
            client.close(timeout=config_data['timeout'])
            cluster.close(timeout=config_data['timeout'])
        except:
            click.echo('Could not restart dask client')
            pass

    return config_data

def apply_pca_wrapper(input_dir, config_data, output_dir, output_file):
    '''
    Wrapper function to obtain PCA Scores.

    Parameters
    ----------
    input_dir (int): path to directory containing all h5+yaml files
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    output_dir (str): path to directory to store PCA data
    output_file (str): pca model filename
    gui (bool): indicate GUI is running

    Returns
    -------
    config_data (dict): updated config_data variable to write back in GUI API
    '''

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    set_dask_config()

    h5s, dicts, yamls = recursive_find_h5s(input_dir)

    output_dir = os.path.abspath(output_dir)

    # automatically get the correct timestamp path
    try:
        h5_timestamp_path = get_timestamp_path(h5s[0])
        h5_metadata_path = get_metadata_path(h5s[0])
    except:
        click.echo('Autoload timestamps failed, will perform search.')

    if config_data['pca_file'] is None:
        pca_file = os.path.join(output_dir, 'pca.h5')
        config_data['pca_file'] = pca_file
    else:
        if not os.path.exists(config_data['pca_file']):
            pca_file = os.path.join(output_dir, 'pca.h5')
            config_data['pca_file'] = pca_file
        else:
            pca_file = config_data['pca_file']

    if not os.path.exists(pca_file):
        raise IOError(f'Could not find PCA components file {pca_file}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_file = os.path.join(output_dir, output_file)

    print('Loading PCs from', pca_file)
    with h5py.File(config_data['pca_file'], 'r') as f:
        pca_components = f[config_data['pca_path']][()]

    # get the yaml for pca, check parameters, if we used fft, be sure to turn on here...
    pca_yaml = os.path.splitext(pca_file)[0] + '.yaml'

    use_fft, clean_params, mask_params, missing_data = get_pca_yaml_data(pca_yaml)

    if use_fft:
        print('Using FFT...')

    with warnings.catch_warnings():
        # warnings.simplefilter("ignore", tqdm.TqdmSynchronisationWarning)
        if config_data['cluster_type'] == 'nodask':
            apply_pca_local(pca_components=pca_components, h5s=h5s, yamls=yamls,
                            use_fft=use_fft, clean_params=clean_params,
                            save_file=save_file, chunk_size=config_data['chunk_size'],
                            mask_params=mask_params, fps=config_data['fps'],
                            missing_data=missing_data, h5_path=config_data['h5_path'],
                            h5_mask_path=config_data['h5_mask_path'])

        else:
            client, cluster, workers = \
                initialize_dask(cluster_type=config_data['cluster_type'],
                                nworkers=config_data['nworkers'],
                                cores=config_data['cores'],
                                processes=config_data['processes'],
                                memory=config_data['memory'],
                                wall_time=config_data['wall_time'],
                                queue=config_data['queue'],
                                timeout=config_data['timeout'],
                                cache_path=config_data['dask_cache_path'],
                                dashboard_port=config_data['dask_port'],
                                data_size=config_data.get('data_size', None))

            logging.basicConfig(filename=f'{output_dir}/scores.log', level=logging.ERROR)

            try:
                apply_pca_dask(pca_components=pca_components, h5s=h5s, yamls=yamls,
                               use_fft=use_fft, clean_params=clean_params,
                               save_file=save_file, chunk_size=config_data['chunk_size'],
                               fps=config_data['fps'], client=client, missing_data=missing_data,
                               mask_params=mask_params, h5_path=config_data['h5_path'],
                               h5_mask_path=config_data['h5_mask_path'])
            except:
                click.echo('Operation interrupted. Closing Dask Client.')
                client.close(timeout=config_data['timeout'])
                cluster.close(timeout=config_data['timeout'])

            if client is not None:
                try:
                    client.close(timeout=config_data['timeout'])
                    cluster.close(timeout=config_data['timeout'])
                except:
                    click.echo('Could not restart dask client')
                    pass


    config_data['pca_file_scores'] = save_file + '.h5'
    return config_data

def compute_changepoints_wrapper(input_dir, config_data, output_dir, output_file):
    '''
    Wrapper function to compute model-free (PCA based) Changepoints.

    Parameters
    ----------
    input_dir (int): path to directory containing all h5+yaml files
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    output_dir (str): path to directory to store PCA data
    output_file (str): pca model filename
    gui (bool): indicate GUI is running

    Returns
    -------
    config_data (dict): updated config_data variable to write back in GUI API
    '''

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    dask_cache_path = os.path.expanduser('~/moseq2_pca')

    config_data, pca_file_components, pca_file_scores, h5s, yamls, save_file = \
        setup_cp_command(input_dir, config_data, output_dir, output_file)

    pca_components, changepoint_params, missing_data, mask_params = \
        load_pcs_for_cp(pca_file_components, config_data)

    client, cluster, workers = \
        initialize_dask(cluster_type=config_data['cluster_type'],
                        nworkers=config_data['nworkers'],
                        cores=config_data['cores'],
                        processes=config_data['processes'],
                        memory=config_data['memory'],
                        wall_time=config_data['wall_time'],
                        queue=config_data['queue'],
                        timeout=config_data['timeout'],
                        cache_path=dask_cache_path,
                        dashboard_port=config_data['dask_port'],
                        data_size=config_data.get('data_size', None))

    logging.basicConfig(filename=f'{output_dir}/changepoints.log', level=logging.ERROR)

    try:
        get_changepoints_dask(pca_components=pca_components, pca_scores=pca_file_scores,
                              h5s=h5s, yamls=yamls, changepoint_params=changepoint_params,
                              save_file=save_file, chunk_size=config_data['chunk_size'],
                              fps=config_data['fps'], client=client, missing_data=missing_data,
                              mask_params=mask_params, h5_path=config_data['h5_path'],
                              h5_mask_path=config_data['h5_mask_path'])
    except:
        click.echo('Operation interrupted. Closing Dask Client.')
        client.close(timeout=config_data['timeout'])

    if client is not None:
        try:
            client.close(timeout=config_data['timeout'])
            cluster.close(timeout=config_data['timeout'])
        except:
            print('Could not restart dask client')
            pass

    import numpy as np
    with h5py.File(f'{save_file}.h5', 'r') as f:
        cps = h5_to_dict(f, 'cps')
    block_durs = np.concatenate([np.diff(cp, axis=0) for k, cp in cps.items()])
    out = changepoint_dist(block_durs, headless=True)
    if out:
        fig, _ = out
        fig.savefig(f'{save_file}_dist.png')
        fig.savefig(f'{save_file}_dist.pdf')
        fig.close('all')


    return config_data