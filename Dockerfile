FROM cuda:v2.0
LABEL maintainer="wadahana <wadahana@gmail.com>"
ENV GRADIO_SERVER_NAME=0.0.0.0

WORKDIR /deep
COPY . /deep
RUN set -ex \
   && chmod a+x /deep/setup.sh \
   && /deep/setup.sh 

#ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/tensorrt_libs:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
