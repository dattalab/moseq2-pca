import warnings
import ruamel.yaml as yaml
from moseq2_pca.helpers.wrappers import train_pca_wrapper, apply_pca_wrapper, compute_changepoints_wrapper

def train_pca_command(input_dir, config_file, output_dir, output_file, output_directory=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    config_data = train_pca_wrapper(input_dir, config_data, output_dir, output_file, output_directory, gui=True)

    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)

def apply_pca_command(input_dir, index_file, config_file, output_dir, output_file, output_directory=None):

    # find directories with .dat files that either have incomplete or no extractions
    # TODO: additional post-processing, intelligent mapping of metadata to group names, make sure
    # moseq2-model processes these files correctly

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    config_data = apply_pca_wrapper(input_dir, config_data, output_dir, output_file, output_directory=output_directory, gui=True)

    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)

    try:
        with open(index_file, 'r') as f:
            index_params = yaml.safe_load(f)
        f.close()

        index_params['pca_path'] = config_data['pca_file_scores']

        with open(index_file, 'w') as f:
            yaml.safe_dump(index_params, f)

        f.close()
    except:
        print('moseq2-index not found, did not update paths')

    return 'PCA Scores have been successfully computed.'


def compute_changepoints_command(input_dir, config_file, output_dir, output_file, output_directory=None):

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    config_data = compute_changepoints_wrapper(input_dir, config_data, output_dir, output_file, gui=True, output_directory=output_directory)

    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)


    return 'Model-free syllable changepoints have been successfully computed.'
