from setuptools import setup

setup(
    name='moseq2-pca',
    author='Jeff Markowitz',
    description='To boldly go where no mouse has gone before',
    version='0.1.0',
    platforms=['mac', 'unix'],
    install_requires=['h5py', 'matplotlib', 'scipy>=0.19',
                      'tqdm', 'numpy', 'joblib==0.11', 'pytest',
                      'opencv-python', 'click', 'ruamel.yaml<=0.15.0',
                      'dask[complete]', 'chest', 'seaborn', 'dask_jobqueue>=0.3.0',
                      'scikit-image>=0.14', 'bokeh'],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['moseq2-pca = moseq2_pca.cli:cli']}
)
