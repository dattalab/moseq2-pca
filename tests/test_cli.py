import pytest
import os
import numpy as np
import h5py
import ruamel.yaml as yaml
from click.testing import CliRunner
from moseq2_pca.cli import train_pca, apply_pca, compute_changepoints


@pytest.fixture(scope='function')
def temp_dir(tmpdir):
    f = tmpdir.mkdir('test_dir')
    return str(f)


def test_train_pca(temp_dir):

    data_path = os.path.join(temp_dir, 'test_mouse.h5')
    yaml_path = os.path.join(temp_dir, 'test_mouse.yaml')

    edge_size = 40
    points = np.arange(-edge_size, edge_size)
    sig1 = 10
    sig2 = 20

    kernel = np.exp(-(points**2.0) / (2.0 * sig1**2.0))
    kernel2 = np.exp(-(points**2.0) / (2.0 * sig2**2.0))

    kernel_full = np.outer(kernel, kernel2)
    kernel_full /= np.max(kernel_full)
    kernel_full *= 50

    fake_mouse = kernel_full
    fake_mouse[fake_mouse < 5] = 0

    fake_movie = np.tile(fake_mouse, (100, 1, 1))

    with h5py.File(data_path, 'w') as f:
        f.create_dataset('frames', data=fake_movie, compression='gzip', dtype='u1')
        f.create_dataset('frames_mask', data=fake_movie > 0, compression='gzip', dtype='bool')

    with open(yaml_path, 'w') as f:
        yaml.dump({'uuid': 'testing'}, f, Dumper=yaml.RoundTripDumper)


    runner = CliRunner()

    result = runner.invoke(train_pca,
                           ['-i', temp_dir, '-o',
                            os.path.join(temp_dir, '_pca'),
                            '-n', 1,
                            '--memory', '5GB',
                            '--visualize-results', True],
                           catch_exceptions=False)

    assert(result.exit_code == 0)

    result = runner.invoke(train_pca,
                           ['-i', temp_dir, '-o',
                            os.path.join(temp_dir, '_pca2'),
                            '-n', 1,
                            '--memory', '5GB',
                            '--missing-data'],
                           catch_exceptions=False)

    assert(result.exit_code == 0)


def test_apply_pca(temp_dir):

    data_path = os.path.join(temp_dir, 'test_mouse.h5')
    yaml_path = os.path.join(temp_dir, 'test_mouse.yaml')
    edge_size = 40
    points = np.arange(-edge_size, edge_size)
    sig1 = 10
    sig2 = 20

    kernel = np.exp(-(points**2.0) / (2.0 * sig1**2.0))
    kernel2 = np.exp(-(points**2.0) / (2.0 * sig2**2.0))

    kernel_full = np.outer(kernel, kernel2)
    kernel_full /= np.max(kernel_full)
    kernel_full *= 50

    fake_mouse = kernel_full
    fake_mouse[fake_mouse < 5] = 0

    fake_movie = np.tile(fake_mouse, (100, 1, 1))

    with h5py.File(data_path, 'w') as f:
        f.create_dataset('frames', data=fake_movie, compression='gzip', dtype='u1')
        f.create_dataset('frames_mask', data=fake_movie > 0, compression='gzip', dtype='bool')
        f.create_dataset('timestamps', data=np.arange(len(fake_movie)) * 30, compression='gzip', dtype='int32')
        f.create_dataset('/metadata/acquisition', data='acquisition')

    with open(yaml_path, 'w') as f:
        yaml.dump({'uuid': 'testing'}, f, Dumper=yaml.RoundTripDumper)

    runner = CliRunner()

    _ = runner.invoke(train_pca,
                      ['-i', temp_dir, '-o',
                       os.path.join(temp_dir, '_pca'),
                       '-n', 1,
                       '--memory', '5GB',
                       '--visualize-results', True],
                      catch_exceptions=False)

    result = runner.invoke(apply_pca,
                           ['-i', temp_dir,
                            '-o', os.path.join(temp_dir, '_pca'),
                            '--pca-file', os.path.join(temp_dir, '_pca/pca.h5'),
                            '--cluster-type', 'nodask'],
                           catch_exceptions=False)

    assert result.exit_code == 0

    result = runner.invoke(apply_pca,
                           ['-i', temp_dir,
                            '-o', os.path.join(temp_dir, '_pca'),
                            '--pca-file', os.path.join(temp_dir, '_pca/pca.h5'),
                            '-n', 1,
                            '--memory', '5GB',
                            '--cluster-type', 'local'],
                           catch_exceptions=False)

    assert result.exit_code == 0

    _ = runner.invoke(train_pca,
                      ['-i', temp_dir, '-o',
                       os.path.join(temp_dir, '_pca2'),
                       '-n', 1,
                       '--memory', '5GB',
                       '--missing-data'],
                      catch_exceptions=False)

    result = runner.invoke(apply_pca,
                           ['-i', temp_dir,
                            '-o', os.path.join(temp_dir, '_pca2'),
                            '--pca-file', os.path.join(temp_dir, '_pca2/pca.h5'),
                            '--cluster-type', 'nodask'],
                           catch_exceptions=False)

    assert result.exit_code == 0

    result = runner.invoke(apply_pca,
                           ['-i', temp_dir,
                            '-o', os.path.join(temp_dir, '_pca2'),
                            '--pca-file', os.path.join(temp_dir, '_pca2/pca.h5'),
                            '-n', 1,
                            '--memory', '5GB',
                            '--cluster-type', 'local'],
                           catch_exceptions=False)

    assert result.exit_code == 0


def test_compute_changepoints(temp_dir):

    data_path = os.path.join(temp_dir, 'test_mouse.h5')
    yaml_path = os.path.join(temp_dir, 'test_mouse.yaml')
    edge_size = 40
    points = np.arange(-edge_size, edge_size)
    sig1 = 10
    sig2 = 20

    kernel = np.exp(-(points**2.0) / (2.0 * sig1**2.0))
    kernel2 = np.exp(-(points**2.0) / (2.0 * sig2**2.0))

    kernel_full = np.outer(kernel, kernel2)
    kernel_full /= np.max(kernel_full)
    kernel_full *= 50

    fake_mouse = kernel_full
    fake_mouse[fake_mouse < 5] = 0

    fake_movie = np.tile(fake_mouse, (100, 1, 1))

    with h5py.File(data_path, 'w') as f:
        f.create_dataset('frames', data=fake_movie, compression='gzip', dtype='u1')
        f.create_dataset('frames_mask', data=fake_movie > 0, compression='gzip', dtype='bool')
        f.create_dataset('timestamps', data=np.arange(len(fake_movie)) * 30, compression='gzip', dtype='int32')
        f.create_dataset('/metadata/acquisition', data='acquisition')

    with open(yaml_path, 'w') as f:
        yaml.dump({'uuid': 'testing'}, f, Dumper=yaml.RoundTripDumper)

    runner = CliRunner()

    _ = runner.invoke(train_pca,
                      ['-i', temp_dir, '-o',
                       os.path.join(temp_dir, '_pca'),
                       '-n', 1,
                       '--memory', '5GB',
                       '--visualize-results', True],
                      catch_exceptions=False)

    _ = runner.invoke(apply_pca,
                      ['-i', temp_dir,
                       '-o', os.path.join(temp_dir, '_pca'),
                       '--pca-file', os.path.join(temp_dir, '_pca/pca.h5'),
                       '--cluster-type', 'nodask'],
                      catch_exceptions=False)

    result = runner.invoke(compute_changepoints,
                           ['-i', temp_dir,
                            '-o', os.path.join(temp_dir, '_pca'),
                            '--pca-file-components', os.path.join(temp_dir, '_pca/pca.h5'),
                            '-n', 1,
                            '--memory', '5GB',
                            '--pca-file-scores', os.path.join(temp_dir, '_pca/pca_scores.h5'),
                            '--cluster-type', 'local'],
                           catch_exceptions=False)

    assert(result.exit_code == 0)

    _ = runner.invoke(train_pca,
                      ['-i', temp_dir, '-o',
                       os.path.join(temp_dir, '_pca2'),
                       '--memory', '5GB',
                       '-n', 1,
                       '--missing-data'],
                      catch_exceptions=False)

    _ = runner.invoke(apply_pca,
                      ['-i', temp_dir,
                       '-o', os.path.join(temp_dir, '_pca2'),
                       '--pca-file', os.path.join(temp_dir, '_pca2/pca.h5'),
                       '--cluster-type', 'nodask'],
                      catch_exceptions=False)

    result = runner.invoke(compute_changepoints,
                           ['-i', temp_dir,
                            '-o', os.path.join(temp_dir, '_pca2'),
                            '--pca-file-components', os.path.join(temp_dir, '_pca2/pca.h5'),
                            '--pca-file-scores', os.path.join(temp_dir, '_pca2/pca_scores.h5'),
                            '-n', 1,
                            '--memory', '5GB',
                            '--cluster-type', 'local'],
                           catch_exceptions=False)

    assert(result.exit_code == 0)
