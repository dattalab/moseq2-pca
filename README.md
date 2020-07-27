# MoSeq2-PCA

[![Build Status](https://travis-ci.com/dattalab/moseq2-pca.svg?token=gvoikVySDHEmvHT7Dbed&branch=test-suite)](https://travis-ci.com/dattalab/moseq2-pca) 

[![codecov](https://codecov.io/gh/dattalab/moseq2-pca/branch/test-suite/graph/badge.svg?token=OLbqEbHHNP)](https://codecov.io/gh/dattalab/moseq2-pca)

This is a library for computing PCA components and scores from extracted mouse movies.  Use this to compute features for modeling.

Latest version is `0.3.0`

## Features
Below are the commands/functionality that moseq2-pca currently affords. 
They are accessible via CLI or Jupyter Notebook in [moseq2-app](https://github.com/dattalab/moseq2-app/tree/release).
```bash
Usage: moseq2-pca [OPTIONS] COMMAND [ARGS]...

Options:
  --help              Show this message and exit.  [default: False]
  --version           Print version number

Commands:
  apply-pca             Computes PCA Scores of extraction data given a...
  clip-scores           Clips speficied number of frames from PCA scores at...
  compute-changepoints  Computes the Model-Free Syllable Changepoints based...
  train-pca             Trains PCA on all extracted results (h5 files) in...

```

Run any command with the `--help` flag to display all available options and their descriptions.

## Documentation

MoSeq2 uses `sphinx` to generate the documentation in HTML and PDF forms. To install `sphinx`, follow the commands below:
```.bash
pip install sphinx==3.0.3
pip install sphinx-rtd-theme
pip install rst2pdf
``` 

All documentation regarding moseq2-extract can be found in the `Documentation.pdf` file in the root directory,
an HTML ReadTheDocs page can be generated via running the `make html` in the `docs/` directory.

To generate a PDF version of the documentation, simply run `make pdf` in the `docs/` directory.

## Prerequisites

In order to use this package you must have already extracted your data via [moseq2-extract](https://github.com/dattalab/moseq2-extract).
If you aggregated your results into a single folder `aggregate_results/`, use that directory as your input directory 
for the `train-pca` command. 

It is also recommended to have also already generated a `moseq2-index.yaml` file to store the path to your `pca_scores` 
file as well.
 - The index file is generated when aggregating the results in [moseq2-extract](https://github.com/dattalab/moseq2-extract/tree/release) 

## Example Outputs

### Mouse Principal Components
<img src="https://github.com/dattalab/moseq2-pca/blob/test-suite/media/Components_Ex.png" width=350 height=350>

### Rat Principal Components
<img src="https://github.com/dattalab/moseq2-pca/blob/test-suite/media/rat_components.png" width=350 height=350>

### Scree Plot
<img src="https://github.com/dattalab/moseq2-pca/blob/test-suite/media/Scree_Ex.png" width=450 height=350>

### Model-free Changepoint Distribution
<img src="https://github.com/dattalab/moseq2-pca/blob/test-suite/media/CP_Ex.png" width=350 height=350>

## Contributing

If you would like to contribute, fork the repository and issue a pull request.
