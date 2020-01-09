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
    version='0.1.3',
    platforms=['mac', 'unix'],
    packages=find_packages(),
    install_requires=['h5py', 'matplotlib', 'scipy>=0.19',
                      'tqdm', 'numpy', 'joblib==0.13.1', 'pytest',
                      'opencv-python', 'click', 'ruamel.yaml>=0.15.0',
                      'dask[complete]', 'chest', 'seaborn', 'dask_jobqueue>=0.3.0',
                      'scikit-image>=0.14', 'bokeh', 'psutil'],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['moseq2-pca = moseq2_pca.cli:cli']}
)
