FROM continuumio/miniconda
ENV PATH /opt/conda/lib:/opt/conda/include:$PATH

# Get a newer build toolchain, sshfs, other stuff
RUN DEBIAN_FRONTEND=noninteractive apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential\
 && apt-get install -y lsb-release\
 && apt-get install -y sshfs\
 && apt-get install -y git

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx

RUN conda create -n "moseq2" python=3.6 -y
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "source activate moseq2" > ~/.bashrc
ENV PATH /opt/conda/envs/moseq2/bin:$PATH
ENV SRC /src
ENV PYTHONPATH /src
RUN mkdir -p $SRC

COPY . $SRC/moseq2-pca
#RUN conda install -c defaults ffmpeg gcc -y
RUN pip install -e $SRC/moseq2-pca
