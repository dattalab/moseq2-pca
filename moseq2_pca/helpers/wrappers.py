'''
Wrapper functions for all functionality included in MoSeq2-PCA that is accessible via CLI or GUI.

Each wrapper function executes the functionality from end-to-end given it's dependency parameters are inputted.
(See CLI Click parameters)
'''

import os
import h5py
import click
import logging
import datetime
import warnings
import dask.array as da
import ruamel.yaml as yaml
from functools import wraps
from moseq2_pca.viz import display_components, scree_plot, changepoint_dist
from moseq2_pca.helpers.data import get_pca_paths, get_pca_yaml_data, load_pcs_for_cp
from moseq2_pca.pca.util import apply_pca_dask, apply_pca_local, train_pca_dask, get_changepoints_dask
from moseq2_pca.util import recursive_find_h5s, select_strel, initialize_dask, set_dask_config, h5_to_dict, get_timestamps

def load_and_check_data(function):
    '''

    Decorator function that executes initialization functionality that is common among all 3 PCA related operations.
    Function will load relevant h5 and yaml files found in given input directory, then check for timestamps and
    warn the user if they are missing.

    Parameters
    ----------
    function (function): train_pca_wrapper, apply_pca_wrapper, compute_changepoints_wrapper

    Returns
    -------
    function (function): decorated function with keyword arguments holding loaded data
    '''

    @wraps(function)
    def wrapped(*args):

        func = function.__name__

        set_dask_config()

        input_dir = args[0]
        output_dir = args[2]

        # Set up output directory
        output_dir = os.path.abspath(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if 'changepoints' in func:
            # Look for aggregated results by default, recursively search for data if aggregate_results path does not exist.
            if os.path.exists(os.path.join(input_dir, 'aggregate_results/')):
                h5s, dicts, yamls = recursive_find_h5s(os.path.join(input_dir, 'aggregate_results/'))
            else:
                h5s, dicts, yamls = recursive_find_h5s(input_dir)
        else:
            # find directories with .dat files that either have incomplete or no extractions
            h5s, dicts, yamls = recursive_find_h5s(input_dir)

        get_timestamps(h5s)  # function to check whether timestamp files are found

        # Save the variables to pass to function
        kwargs = {
            'h5s': h5s,
            'yamls': yamls,
            'dicts': dicts
        }

        return function(*args, **kwargs)
    return wrapped

@load_and_check_data
def train_pca_wrapper(input_dir, config_data, output_dir, output_file, **kwargs):
    '''
    Wrapper function to train PCA.

    Note: function is decorated with function performing initialization operations and saving
    the results in the kwargs variable.

    Parameters
    ----------
    input_dir (int): path to directory containing all h5+yaml files
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    output_dir (str): path to directory to store PCA data
    output_file (str): pca model filename
    kwargs (dict): dictionary containing loaded h5s, yamls and dicts found in given input_dir

    Returns
    -------
    config_data (dict): updated config_data variable to write back in GUI API
    '''

    if config_data['missing_data'] and config_data['use_fft']:
        raise NotImplementedError("FFT and missing data not implemented yet")

    params = config_data

    h5s = kwargs['h5s']

    params['start_time'] = f'{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}'
    params['inputs'] = h5s

    # Setting path to PCA config file
    save_file = os.path.join(output_dir, output_file)

    # Edge Case: Handling pre-existing PCA file
    if os.path.exists(f'{save_file}.h5'):
        click.echo(f'The file {save_file}.h5 already exists.\nWould you like to overwrite it? [y -> yes, else -> exit]\n')
        ow = input()
        if ow.lower() != 'y':
            return config_data

    # Update PCA config yaml file
    config_store = '{}.yaml'.format(save_file)
    with open(config_store, 'w') as f:
        yaml.safe_dump(params, f)

    # Hold all frame filtering parameters in a single dict
    clean_params = {
        'gaussfilter_space': config_data['gaussfilter_space'],
        'gaussfilter_time': config_data['gaussfilter_time'],
        'tailfilter': select_strel((config_data['tailfilter_shape'], config_data['tailfilter_size'])),
        'medfilter_time': config_data['medfilter_time'],
        'medfilter_space': config_data['medfilter_space']
    }

    logging.basicConfig(filename=f'{output_dir}/train.log', level=logging.ERROR)

    # Load all h5 file references to extracted frames, then read them into chunked Dask arrays
    dsets = [h5py.File(h5, mode='r')[config_data['h5_path']] for h5 in h5s]
    arrays = [da.from_array(dset, chunks=config_data['chunk_size']) for dset in dsets]
    stacked_array = da.concatenate(arrays, axis=0)

    # Filter out depth value optimas; Generally same values used during extraction
    stacked_array[stacked_array < config_data['min_height']] = 0
    stacked_array[stacked_array > config_data['max_height']] = 0

    config_data['data_size'] = stacked_array.nbytes

    # Initialize Dask client
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

    # Optionally read corresponding frame masks if recomputing PC scores for dropped frames
    # Note: timestamps for all files are required in order for this operation to work.
    if config_data['missing_data']:
        mask_dsets = [h5py.File(h5, mode='r')[config_data['h5_mask_path']] for h5 in h5s]
        mask_arrays = [da.from_array(dset, chunks=config_data['chunk_size']) for dset in mask_dsets]
        stacked_array_mask = da.concatenate(mask_arrays, axis=0).astype('float32')
        stacked_array_mask = da.logical_and(stacked_array_mask < config_data['mask_threshold'],
                                            stacked_array > config_data['mask_height_threshold'])
        click.echo('Loaded mask for missing data')

    else:
        stacked_array_mask = None

    # Compute Principal Components
    try:
        output_dict = \
            train_pca_dask(dask_array=stacked_array, mask=stacked_array_mask,
                           clean_params=clean_params, use_fft=config_data['use_fft'],
                           rank=config_data['rank'], cluster_type=config_data['cluster_type'],
                           min_height=config_data['min_height'],
                           max_height=config_data['max_height'], client=client,
                           iters=config_data['missing_data_iters'],
                           recon_pcs=config_data['recon_pcs'])
    except Exception as e:
        # Clearing all data from Dask client in case of interrupted PCA
        logging.error(e)
        logging.error(e.__traceback__)
        click.echo('Training interrupted. Closing Dask Client. You may find logs of the error here:')
        click.echo('---- ', os.path.join(output_dir, 'train.log'))
        client.close(timeout=config_data['timeout'])
        cluster.close(timeout=config_data['timeout'])

    # Plotting PCs
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

    # Plotting Scree Plot
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

    # Saving PCA to h5 file
    with h5py.File(f'{save_file}.h5', 'w') as f:
        for k, v in output_dict.items():
            f.create_dataset(k, data=v, compression='gzip', dtype='float32')

    config_data['pca_file'] = f'{save_file}.h5'

    # After Success: Shutting down Dask client and clearing any residual data
    if client is not None:
        try:
            client.close(timeout=config_data['timeout'])
            cluster.close(timeout=config_data['timeout'])
        except:
            click.echo('Could not restart dask client')
            pass

    return config_data

@load_and_check_data
def apply_pca_wrapper(input_dir, config_data, output_dir, output_file, **kwargs):
    '''
    Wrapper function to obtain PCA Scores.

    Parameters
    ----------
    input_dir (int): path to directory containing all h5+yaml files
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    output_dir (str): path to directory to store PCA data
    output_file (str): pca model filename
    gui (bool): indicate GUI is running
    kwargs (dict): dictionary containing loaded h5s, yamls and dicts found in given input_dir

    Note: function is decorated with function performing initialization operations and saving
    the results in the kwargs variable.

    Returns
    -------
    config_data (dict): updated config_data variable to write back in GUI API
    '''

    # TODO: additional post-processing, intelligent mapping of metadata to group names, make sure

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    h5s = kwargs['h5s']
    yamls = kwargs['yamls']

    # Set path to PCA Scores file
    save_file = os.path.join(output_dir, output_file)

    # Get path to trained PCA file to load PCs from
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

    print('Loading PCs from', pca_file)
    with h5py.File(config_data['pca_file'], 'r') as f:
        pca_components = f[config_data['pca_path']][()]

    # Get the yaml for pca, check parameters, if we used fft, be sure to turn on here...
    pca_yaml = os.path.splitext(pca_file)[0] + '.yaml'

    # Get filtering parameters and optional PCA reconstruction parameters (if missing_data == True)
    use_fft, clean_params, mask_params, missing_data = get_pca_yaml_data(pca_yaml)

    with warnings.catch_warnings():
        # Compute PCA Scores locally (without dask)
        if config_data['cluster_type'] == 'nodask':
            apply_pca_local(pca_components=pca_components, h5s=h5s, yamls=yamls,
                            use_fft=use_fft, clean_params=clean_params,
                            save_file=save_file, chunk_size=config_data['chunk_size'],
                            mask_params=mask_params, fps=config_data['fps'],
                            missing_data=missing_data, h5_path=config_data['h5_path'],
                            h5_mask_path=config_data['h5_mask_path'])

        else:
            # Initialize Dask client
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

            # Compute PCA Scores
            try:
                apply_pca_dask(pca_components=pca_components, h5s=h5s, yamls=yamls,
                               use_fft=use_fft, clean_params=clean_params,
                               save_file=save_file, chunk_size=config_data['chunk_size'],
                               fps=config_data['fps'], client=client, missing_data=missing_data,
                               mask_params=mask_params, h5_path=config_data['h5_path'],
                               h5_mask_path=config_data['h5_mask_path'])
            except:
                # Clearing all data from Dask client in case of interrupted PCA
                click.echo('Operation interrupted. Closing Dask Client.')
                client.close(timeout=config_data['timeout'])
                cluster.close(timeout=config_data['timeout'])

            # After Success: Shutting down Dask client and clearing any residual data
            if client is not None:
                try:
                    client.close(timeout=config_data['timeout'])
                    cluster.close(timeout=config_data['timeout'])
                except:
                    click.echo('Could not restart dask client')
                    pass


    config_data['pca_file_scores'] = save_file + '.h5'
    return config_data

@load_and_check_data
def compute_changepoints_wrapper(input_dir, config_data, output_dir, output_file, **kwargs):
    '''
    Wrapper function to compute model-free (PCA based) Changepoints.

    Note: function is decorated with function performing initialization operations and saving
    the results in the kwargs variable.

    Parameters
    ----------
    input_dir (int): path to directory containing all h5+yaml files
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    output_dir (str): path to directory to store PCA data
    output_file (str): pca model filename
    gui (bool): indicate GUI is running
    kwargs (dict): dictionary containing loaded h5s, yamls and dicts found in given input_dir

    Returns
    -------
    config_data (dict): updated config_data variable to write back in GUI API
    '''

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Get loaded h5s and yamls
    h5s, yamls, dicts = kwargs['h5s'], kwargs['yamls'], kwargs['dicts']

    dask_cache_path = os.path.expanduser('~/moseq2_pca')

    # Set path to changepoints
    save_file = os.path.join(output_dir, output_file)

    # Get paths to PCA, PCA Scores file
    config_data, pca_file_components, pca_file_scores = get_pca_paths(config_data, output_dir)

    # Load Principal components, set up changepoint parameter dict, and optionally load reconstructed PCs.
    pca_components, changepoint_params, missing_data, mask_params = \
        load_pcs_for_cp(pca_file_components, config_data)

    # Initialize Dask client
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

    # Compute Changepoints
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

    # After Success: Shutting down Dask client and clearing any residual data
    if client is not None:
        try:
            client.close(timeout=config_data['timeout'])
            cluster.close(timeout=config_data['timeout'])
        except:
            print('Could not restart dask client')
            pass

    # Write Changepoints to save file
    import numpy as np
    with h5py.File(f'{save_file}.h5', 'r') as f:
        cps = h5_to_dict(f, 'cps')

    # Plot and save Changepoint PDF histogram
    block_durs = np.concatenate([np.diff(cp, axis=0) for k, cp in cps.items()])
    out = changepoint_dist(block_durs, headless=True)
    if out:
        fig, _ = out
        fig.savefig(f'{save_file}_dist.png')
        fig.savefig(f'{save_file}_dist.pdf')
        fig.close('all')

    return config_data