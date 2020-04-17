from setuptools import setup, find_packages
import subprocess
import sys

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


try:
    import cv2
except ImportError:
    install('opencv-python')


setup(
    name='moseq2-pca',
    author='Jeff Markowitz',
    description='To boldly go where no mouse has gone before',
    version='0.5.1',
    platforms=['mac', 'unix'],
    packages=find_packages(),
    install_requires=['h5py', 'dask', 'tqdm', 'bokeh', 'chest', 'scipy',
                      'numpy', 'click', 'psutil', 'joblib', 'cytoolz', 'matplotlib',
                      'statsmodels', 'ruamel.yaml', 'distributed', 'scikit-image',
                      'scikit-learn', 'dask.jobqueue', 'opencv-python'],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['moseq2-pca = moseq2_pca.cli:cli']}
)
