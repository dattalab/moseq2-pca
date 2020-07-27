import os
import h5py
import ruamel.yaml as yaml
from moseq2_pca.util import recursive_find_h5s, select_strel, get_timestamps

def setup_cp_command(input_dir, config_data, output_dir, output_file):
    '''
    Helper function for changepoints_wrapper to perform data-path existence checks.

    Parameters
    ----------
    input_dir (int): path to directory containing all h5+yaml files
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    output_dir (str): path to directory to store PCA data
    output_file (str): pca model filename

    Returns
    -------
    config_data (dict): updated config_data dict with the proper paths
    pca_file_components (str): path to trained pca file
    pca_file_scores (str): path to pca_scores file
    h5s (list): list of relevant pca h5 files
    yamls (list): list of relevant pca metadata yaml files
    save_file (str): path to save changepoints
    '''

    if os.path.exists(os.path.join(input_dir, 'aggregate_results/')):
        h5s, dicts, yamls = recursive_find_h5s(os.path.join(input_dir, 'aggregate_results/'))
    else:
        h5s, dicts, yamls = recursive_find_h5s(input_dir)

    get_timestamps(h5s) # function to check whether timestamp files are found

    output_dir = os.path.abspath(output_dir)

    if config_data.get('pca_file_components') is None:
        pca_file_components = os.path.join(output_dir, 'pca.h5')
        config_data['pca_file_components'] = pca_file_components
    else:
        if not os.path.exists(config_data['pca_file_components']):
            pca_file_components = os.path.join(output_dir, 'pca.h5')
            config_data['pca_file_components'] = pca_file_components
        else:
            pca_file_components = config_data['pca_file_components']

    if config_data.get('pca_file_scores') is None:
        pca_file_scores = os.path.join(output_dir, 'pca_scores.h5')
        config_data['pca_file_scores'] = pca_file_scores
    else:
        pca_file_scores = config_data['pca_file_scores']

    if not os.path.exists(pca_file_components):
        raise IOError(f'Could not find PCA components file {pca_file_components}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_file = os.path.join(output_dir, output_file)

    return config_data, pca_file_components, pca_file_scores, h5s, yamls, save_file

def load_pcs_for_cp(pca_file_components, config_data):
    '''
    Load computed Principal Components for Model-free Changepoint Analysis.

    Parameters
    ----------
    pca_file_components (str): path to pca h5 file to read PCs
    config_data (dict): config parameters

    Returns
    -------
    pca_components (str): path to pca components
    changepoint_params (dict): dict of relevant changepoint parameters
    cluster (dask Cluster): Dask Cluster object.
    client (dask Client): Dask Client Object
    workers (dask Workers): intialized workers or None if cluster_type = 'local'
    missing_data (bool): Indicates whether to use mask_params
    mask_params (dict): Mask parameters to use when computing CPs
    '''

    print('Loading PCs from {}'.format(pca_file_components))
    with h5py.File(pca_file_components, 'r') as f:
        pca_components = f[config_data['pca_path']][...]

    # get the yaml for pca, check parameters, if we used fft, be sure to turn on here...
    pca_yaml = os.path.splitext(pca_file_components)[0] + '.yaml'

    # TODO: Detect missing data and mask parameters, then 0 out, fill in, compute scores...
    if os.path.exists(pca_yaml):
        with open(pca_yaml, 'r') as f:
            pca_config = yaml.safe_load(f.read())

            if 'missing_data' in pca_config.keys() and pca_config['missing_data']:
                print('Detected missing data...')
                missing_data = True
                mask_params = {
                    'mask_height_threshold': pca_config['mask_height_threshold'],
                    'mask_threshold': pca_config['mask_threshold']
                }
            else:
                missing_data = False
                mask_params = None

            if missing_data and not os.path.exists(config_data['pca_file_scores']):
                raise RuntimeError("Need PCA scores to impute missing data, run apply pca first")

    changepoint_params = {
        'k': config_data['klags'],
        'sigma': config_data['sigma'],
        'peak_height': config_data['threshold'],
        'peak_neighbors': config_data['neighbors'],
        'rps': config_data['dims']
    }

    return pca_components, changepoint_params, missing_data, mask_params

def get_pca_yaml_data(pca_yaml):
    '''
    Reads PCA yaml file and returns metadata

    Parameters
    ----------
    pca_yaml (str): path to pca.yaml

    Returns
    -------
    use_fft (bool): indicates whether to use FFT
    clean_params (dict): dict of image filtering parameters
    mask_params (dict): dict of mask parameters)
    missing_data (bool): indicates whether to use mask_params
    '''

    # todo detect missing data and mask parameters, then 0 out, fill in, compute scores...
    if os.path.exists(pca_yaml):
        with open(pca_yaml, 'r') as f:
            pca_config = yaml.safe_load(f.read())
            if 'use_fft' in pca_config.keys() and pca_config['use_fft']:
                print('Will use FFT...')
                use_fft = True
            else:
                use_fft = False

            tailfilter = select_strel(pca_config['tailfilter_shape'],
                                      tuple(pca_config['tailfilter_size']))

            clean_params = {
                'gaussfilter_space': pca_config['gaussfilter_space'],
                'gaussfilter_time': pca_config['gaussfilter_time'],
                'tailfilter': tailfilter,
                'medfilter_time': pca_config['medfilter_time'],
                'medfilter_space': pca_config['medfilter_space'],
            }

            mask_params = {
                'mask_height_threshold': pca_config['mask_height_threshold'],
                'mask_threshold': pca_config['mask_threshold'],
                'min_height': pca_config['min_height'],
                'max_height': pca_config['max_height']
            }

            if 'missing_data' in pca_config.keys() and pca_config['missing_data']:
                print('Detected missing data...')
                missing_data = True
            else:
                missing_data = False

    else:
        IOError(f'Could not find {pca_yaml}')

    return use_fft, clean_params, mask_params, missing_data