# dimension-reduction

This is a library for computing PCA components and scores from extracted mouse movies as part of the MoSeq pipeline. Use this to compute features for modeling.

# Pulling an existing container
Make sure that your machine is authorized to access our AWS, using `aws configure`. Then authenticate with Amazon ECR (Elastic Container Registry), where we store built containers.
```
$(aws ecr get-login --region us-east-2 --no-include-email)
docker pull 365166883642.dkr.ecr.us-east-2.amazonaws.com/dimension-reduction:latest
docker tag -t dimension-reduction:latest 365166883642.dkr.ecr.us-east-2.amazonaws.com/dimension-reduction:latest  # for convenience
```

# Build the container yourself
```
git clone https://github.com/syllable-life-sciences/dimension-reduction.git
cd dimension-reduction
docker build -t dimension-reduction .
```

# Running a dimension-reduction container
The general use of `dimension-reduction` is based on the `train-pca` and `apply-pca` commands. The syntax for their use can be found in the docstrings, but the base uses are:

```
moseq2-pca train-pca -i extracted_data_dir -o output_dir
```

```
moseq2-pca apply-pca -i extracted_data_dir -o output_dir
```

The final output of this section of the pipeline is an `h5` file filled with the calculated principal components. The default name is `pca_scores.h5`. Typical use of `dimension-reduction` is done with Docker, though it can be installed and run locally as well.

## Docker usage of dimension-reduction
Using the `-v` flag with Docker, we can attach a local directory to the container so that it can actually operate on files.
```
docker container run -v /local/path/to/data_directory:/data -t dimension-reduction moseq2-pca train-pca -i /data/extracted_data_dir -o /data/_pca
```
Typical usage includes the flip classifier, which properly orients the mouse in each frame so that it is consistently facing the same direction. The classifier is packaged with dimension-reduction, so you can just do:
```
docker container run -v /local/path/to/data_directory:/data -it dimension-reduction moseq2-pca apply-pca -i /data/extracted_data_dir -o /data/_pca
```
