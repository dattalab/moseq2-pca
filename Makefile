.PHONY: data

data: data/_pca/pca.h5 data/_pca/pca.yaml data/_pca/pca_scores.h5 data/_pca/pca_scores.yaml data/proc/results_00.h5 \
data/proc/results_00.yaml data/proc/roi_00.tiff data/pca_test.yaml data/config.yaml data/test_index.yaml \
data/test_scores.h5 data/testh5.h5

data/_pca/pca.h5:
	aws s3 cp s3://moseq2-testdata/pca/_pca/ data/_pca/ --request-payer=requester --recursive

data/_pca/pca.yaml:
	aws s3 cp s3://moseq2-testdata/pca/_pca/ data/_pca/ --request-payer=requester --recursive

data/_pca/pca_scores.h5:
	aws s3 cp s3://moseq2-testdata/pca/_pca/ data/_pca/ --request-payer=requester --recursive

data/_pca/pca_scores.yaml:
	aws s3 cp s3://moseq2-testdata/pca/_pca/ data/_pca/ --request-payer=requester --recursive

data/proc/results_00.h5:
	aws s3 cp s3://moseq2-testdata/pca/proc/ data/proc/ --request-payer=requester --recursive

data/proc/results_00.yaml:
	aws s3 cp s3://moseq2-testdata/pca/proc/ data/proc/ --request-payer=requester --recursive

data/proc/roi_00.tiff:
	aws s3 cp s3://moseq2-testdata/pca/proc/ data/proc/ --request-payer=requester --recursive

data/pca_test.yaml:
	aws s3 cp s3://moseq2-testdata/pca/ data/ --request-payer=requester --recursive

data/config.yaml:
	aws s3 cp s3://moseq2-testdata/pca/ data/ --request-payer=requester --recursive

data/test_index.yaml:
	aws s3 cp s3://moseq2-testdata/pca/ data/ --request-payer=requester --recursive

data/test_scores.h5:
	aws s3 cp s3://moseq2-testdata/pca/ data/ --request-payer=requester --recursive

data/testh5.h5:
	aws s3 cp s3://moseq2-testdata/pca/ data/ --request-payer=requester --recursive