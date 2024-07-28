FROM debian

USER root

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y curl

RUN curl -O https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.7.2-linux-x64.tar.gz && tar xfz dorado-0.7.2-linux-x64.tar.gz && rm dorado-0.7.2-linux-x64.tar.gz

# Note: Install Nvidia runtime for your docker before running this container, check 
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html
# Run it like:
# docker run -it --runtime=nvidia --gpus all -v /MY_PATH:/ftmp CONTAINER_NAME 
# Sample:
# docker run --rm --runtime=nvidia --gpus all -v /MY_PATH:/ftmp CONTAINER_NAME bash -c "dorado-0.7.2-linux-x64/bin/dorado basecaller hac /ftmp/ > /ftmp/out.bam"
