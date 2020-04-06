import os
import h5py
import tqdm
import click
import pathlib
from moseq2_pca.util import command_with_config
from moseq2_pca.helpers.wrappers import train_pca_wrapper, apply_pca_wrapper, compute_changepoints_wrapper

orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init


@click.group()
def cli():
    pass


@cli.command('clip-scores')
@click.argument('pca_file', type=click.Path(exists=True, resolve_path=True))
@click.argument('clip_samples', type=int)
@click.option('--from-end', type=bool, is_flag=True)
def clip_scores(pca_file, clip_samples, from_end):
    """
    Clips PCA scores from the beginning or end

    Args:
        pca_file (string): Path to PCA scores
        clip_samples (int): number of samples to clip from beginning or end
        from_end (bool): if true clip from end rather than beginning

    Note that scores are modified *in place*.
    """

    with h5py.File(pca_file, 'r') as f:
        store_dir = os.path.dirname(pca_file)
        base_filename = os.path.splitext(os.path.basename(pca_file))[0]
        new_filename = os.path.join(store_dir, '{}_clip.h5'.format(base_filename))

        with h5py.File(new_filename, 'w') as f2:
            f.copy('/metadata', f2)
            for key in tqdm.tqdm(f['/scores'].keys(), desc='Copying data'):
                if from_end:
                    f2['/scores/{}'.format(key)] = f['/scores/{}'.format(key)][:-clip_samples]
                    f2['/scores_idx/{}'.format(key)] = f['/scores_idx/{}'.format(key)][:-clip_samples]
                else:
                    f2['/scores/{}'.format(key)] = f['/scores/{}'.format(key)][clip_samples:]
                    f2['/scores_idx/{}'.format(key)] = f['/scores_idx/{}'.format(key)][clip_samples:]

@cli.command(name='train-pca', cls=command_with_config('config_file'))
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--cluster-type', type=click.Choice(['local', 'slurm']),
              default='local', help='Cluster type')
@click.option('--output-dir', '-o', default=os.path.join(os.getcwd(), '_pca'), type=click.Path(), help='Directory to store results')
@click.option('--gaussfilter-space', default=(1.5, 1), type=(float, float), help="Spatial filter for data (Gaussian)")
@click.option('--gaussfilter-time', default=0, type=float, help="Temporal filter for data (Gaussian)")
@click.option('--medfilter-space', default=[0], type=int, help="Median spatial filter", multiple=True)
@click.option('--medfilter-time', default=[0], type=int, help="Median temporal filter", multiple=True)
@click.option('--missing-data', is_flag=True, type=bool, help="Use missing data PCA")
@click.option('--missing-data-iters', default=10, type=int, help="Missing data PCA iterations")
@click.option('--mask-threshold', default=-16, type=float, help="Threshold for mask (missing data only)")
@click.option('--mask-height-threshold', default=5, type=float, help="Threshold for mask based on floor height")
@click.option('--min-height', default=10, type=int, help='Min mouse height from floor (mm)')
@click.option('--max-height', default=100, type=int, help='Max mouse height from floor (mm)')
@click.option('--tailfilter-size', default=(9, 9), type=(int, int), help='Tail filter size')
@click.option('--tailfilter-shape', default='ellipse', type=str, help='Tail filter shape')
@click.option('--use-fft', type=bool, is_flag=True, help='Use 2D fft')
@click.option('--recon-pcs', type=int, default=10, help='Number of PCs to use for missing data reconstruction')
@click.option('--rank', default=50, type=int, help="Rank for compressed SVD (generally>>nPCS)")
@click.option('--output-file', default='pca', type=str, help='Name of h5 file for storing pca results')
@click.option('--chunk-size', default=4000, type=int, help='Number of frames per chunk')
@click.option('--visualize-results', default=True, type=bool, help='Visualize results')
@click.option('--config-file', type=click.Path(), help="Path to configuration file")
@click.option('--dask-cache-path', '-d', default=os.path.join(pathlib.Path.home(), 'moseq2_pca'), type=click.Path(), help='Path to spill data to disk for dask local scheduler')
@click.option('--local-processes', default=True, type=bool, help='Use processes with local scheduler')
@click.option('-q', '--queue', type=str, default='debug', help="Cluster queue/partition for submitting jobs")
@click.option('-n', '--nworkers', type=int, default=10, help="Number of workers")
@click.option('-c', '--cores', type=int, default=1, help="Number of cores per worker")
@click.option('-p', '--processes', type=int, default=1, help="Number of processes to run on each worker")
@click.option('-m', '--memory', type=str, default="15GB", help="Total RAM usage per worker")
@click.option('-w', '--wall-time', type=str, default="06:00:00", help="Wall time for workers")
@click.option('--timeout', type=float, default=5, help="Time to wait for workers to initialize before proceeding (minutes)")
def train_pca(input_dir, cluster_type, output_dir, gaussfilter_space,
              gaussfilter_time, medfilter_space, medfilter_time, missing_data, missing_data_iters, mask_threshold, mask_height_threshold, min_height, max_height, tailfilter_size,
              tailfilter_shape, use_fft, recon_pcs, rank, output_file, chunk_size,
              visualize_results, config_file, dask_cache_path, local_processes, queue, nworkers,
              cores, processes, memory, wall_time, timeout):

    click_data = click.get_current_context().params
    cli_data = {k: v for k, v in click_data.items()}

    train_pca_wrapper(input_dir, cli_data, output_dir, output_file)


@cli.command(name='apply-pca', cls=command_with_config('config_file'))
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--cluster-type', type=click.Choice(['local', 'slurm', 'nodask']),
              default='local', help='Cluster type')
@click.option('--output-dir', '-o', default=os.path.join(os.getcwd(), '_pca'), type=click.Path(exists=False), help='Directory to store results')
@click.option('--output-file', default='pca_scores', type=str, help='Name of h5 file for storing pca results')
@click.option('--h5-path', default='/frames', type=str, help='Path to data in h5 files')
@click.option('--h5-mask-path', default='/frames_mask', type=str, help="Path to log-likelihood mask in h5 files")
@click.option('--pca-path', default='/components', type=str, help='Path to pca components')
@click.option('--pca-file', default=None, type=click.Path(), help='Path to PCA results')
@click.option('--chunk-size', default=4000, type=int, help='Number of frames per chunk')
@click.option('--fill-gaps', default=True, type=bool, help='Fill dropped frames with nans')
@click.option('--fps', default=30, type=int, help='Fps (only used if no timestamps found)')
@click.option('--detrend-window', default=0, type=float, help="Length of detrend window (in seconds, 0 for no detrending)")
@click.option('--config-file', type=click.Path(), help="Path to configuration file")
@click.option('--dask-cache-path', '-d', default=os.path.join(pathlib.Path.home(), 'moseq2_pca'), type=click.Path(), help='Path to spill data to disk for dask local scheduler')
@click.option('-q', '--queue', type=str, default='debug', help="Cluster queue/partition for submitting jobs")
@click.option('-n', '--nworkers', type=int, default=10, help="Number of workers")
@click.option('-c', '--cores', type=int, default=1, help="Number of cores per worker")
@click.option('-p', '--processes', type=int, default=1, help="Number of processes to run on each worker")
@click.option('-m', '--memory', type=str, default="15GB", help="RAM usage per workers")
@click.option('-w', '--wall-time', type=str, default="06:00:00", help="Wall time for workers")
@click.option('--timeout', type=float, default=5, help="Time to wait for workers to initialize before proceeding (minutes)")
def apply_pca(input_dir, cluster_type, output_dir, output_file, h5_path, h5_mask_path,
              pca_path, pca_file, chunk_size, fill_gaps, fps, detrend_window,
              config_file, dask_cache_path, queue, nworkers, cores, processes, memory, wall_time, timeout):

    click_data = click.get_current_context().params
    cli_data = {k: v for k, v in click_data.items()}
    apply_pca_wrapper(input_dir, cli_data, output_dir, output_file)

@cli.command('compute-changepoints', cls=command_with_config('config_file'))
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--output-dir', '-o', default=os.path.join(os.getcwd(), '_pca/'), type=click.Path(exists=False), help='Directory to store results')
@click.option('--output-file', default='changepoints', type=str, help='Name of h5 file for storing pca results')
@click.option('--cluster-type', type=click.Choice(['local', 'slurm']), default='local', help='Cluster type')
@click.option('--pca-file-components', type=click.Path(), default=None, help="Path to PCA components")
@click.option('--pca-file-scores', type=click.Path(), default=None, help='Path to PCA results')
@click.option('--pca-path', default='/components', type=str, help='Path to pca components')
@click.option('--neighbors', type=int, default=1, help="Neighbors to use for peak identification")
@click.option('--threshold', type=float, default=.5, help="Peak threshold to use for changepoints")
@click.option('-k', '--klags', type=int, default=6, help="Lag to use for derivative calculation")
@click.option('-s', '--sigma', type=float, default=3.5, help="Standard deviation of gaussian smoothing filter")
@click.option('-d', '--dims', type=int, default=300, help="Number of random projections to use")
@click.option('--fps', default=30, type=int, help='Fps (only used if no timestamps found)')
@click.option('--h5-path', default='/frames', type=str, help='Path to data in h5 files')
@click.option('--h5-mask-path', default='/frames_mask', type=str, help="Path to log-likelihood mask in h5 files")
@click.option('--chunk-size', default=4000, type=int, help='Number of frames per chunk')
@click.option('--config-file', type=click.Path(), help="Path to configuration file")
@click.option('--dask-cache-path', default=os.path.join(pathlib.Path.home(), 'moseq2_pca'), type=click.Path(), help='Path to spill data to disk for dask local scheduler')
@click.option('--visualize-results', default=True, type=bool, help='Visualize results')
@click.option('-q', '--queue', type=str, default='debug', help="Cluster queue/partition for submitting jobs")
@click.option('-n', '--nworkers', type=int, default=10, help="Number of workers")
@click.option('-c', '--cores', type=int, default=1, help="Number of cores per worker")
@click.option('-p', '--processes', type=int, default=1, help="Number of processes to run on each worker")
@click.option('-m', '--memory', type=str, default="15GB", help="RAM usage per workers")
@click.option('-w', '--wall-time', type=str, default="06:00:00", help="Wall time for workers")
@click.option('--timeout', type=float, default=5, help="Time to wait for workers to initialize before proceeding (minutes)")
def compute_changepoints(input_dir, output_dir, output_file, cluster_type, pca_file_components,
                         pca_file_scores, pca_path, neighbors, threshold, klags, sigma, dims, fps, h5_path,
                         h5_mask_path, chunk_size, config_file, dask_cache_path,
                         visualize_results, queue, nworkers, cores, processes, memory, wall_time, timeout):

    click_data = click.get_current_context().params
    cli_data = {k: v for k, v in click_data.items()}
    compute_changepoints_wrapper(input_dir, cli_data, output_dir, output_file)

if __name__ == '__main__':
    cli()
