'''

GUI front-end operations. This module contains all the functionality and configurable parameters
users can alter to most accurately process their data.

Note: These functions perform jupyter notebook specific preprocessing, loads in corresponding parameters from the
CLI functions, then call the corresponding wrapper function with the given input parameters.

'''

import warnings
import ruamel.yaml as yaml
from .cli import train_pca, apply_pca, compute_changepoints
from moseq2_pca.util import read_yaml
from moseq2_pca.helpers.wrappers import train_pca_wrapper, apply_pca_wrapper, compute_changepoints_wrapper


def train_pca_command(progress_paths, output_dir, output_file):
    '''
    Train PCA through Jupyter notebook, and updates config file.

    Parameters
    ----------
    progress_paths (dict): dictionary containing notebook progress paths
    output_dir (str): path to output pca directory
    output_file (str): name of output pca file.

    Returns
    -------
    '''
    # Get default CLI params
    default_params = {tmp.name: tmp.default for tmp in train_pca.params if not tmp.required}

    # Get appropriate inputs
    input_dir = progress_paths['train_data_dir']
    config_file = progress_paths['config_file']

    config_data = read_yaml(config_file)
    # merge default params with those in config
    config_data = {**default_params, **config_data}

    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        config_data = train_pca_wrapper(input_dir, config_data, output_dir, output_file)

    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)


def apply_pca_command(progress_paths, output_file):
    '''
    Compute PCA Scores given trained PCA using Jupyter Notebook.

    Parameters
    ----------
    progress_paths (dict): dictionary containing notebook progress paths
    output_file (str): name of output pca file.

    Returns
    -------
    (str): success string.
    '''

    # Get default CLI params
    default_params = {tmp.name: tmp.default for tmp in apply_pca.params if not tmp.required}

    # Get proper inputs
    input_dir = progress_paths['train_data_dir']
    config_file = progress_paths['config_file']
    index_file = progress_paths['index_file']
    output_dir = progress_paths['pca_dirname']

    config_data = read_yaml(config_file)
    # merge default params with those in config
    config_data = {**default_params, **config_data}

    config_data = apply_pca_wrapper(input_dir, config_data, output_dir, output_file)

    success_str = ''

    index_params = read_yaml(index_file)
    if index_params:
        if config_data.get('overwrite_pca_apply', False):
            index_params['pca_path'] = config_data['pca_file_scores']
            success_str = 'PCA Scores have been successfully computed.'
        else:
            config_data.pop('overwrite_pca_apply', None)
            success_str = 'Preexisting PCA Scores have not been updated.'

        with open(index_file, 'w') as f:
            yaml.safe_dump(index_params, f)

    else:
        print('moseq2-index not found, did not update paths')

    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)

    return success_str


def compute_changepoints_command(input_dir, progress_paths, output_file):
    '''
    Compute Changepoint distribution using Jupyter Notebook.

    Parameters
    ----------
    input_dir (str): path to directory containing training data
    progress_paths (dict): dictionary containing notebook progress paths
    output_file (str): name of output pca file.

    Returns
    -------
    (str): success string.
    '''
    # Get default CLI params
    default_params = {tmp.name: tmp.default for tmp in compute_changepoints.params
                      if not tmp.required}

    config_file = progress_paths['config_file']
    output_dir = progress_paths['pca_dirname']

    config_data = read_yaml(config_file)
    # merge default params with those in config
    config_data = {**default_params, **config_data}

    config_data = compute_changepoints_wrapper(input_dir, config_data, output_dir, output_file)

    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)

    return 'Model-free syllable changepoints have been successfully computed.'
