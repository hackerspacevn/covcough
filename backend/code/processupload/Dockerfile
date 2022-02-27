# Pull the base image with python 3.7 as a runtime for your Lambda to solve libsnd issue
FROM public.ecr.aws/lambda/python:3.7
ENV PYTHON_VERSION=3.7.7
ENV SNDFILE_VERSION=1.0.28

RUN yum install -y tar xz yum-utils

WORKDIR /tmp

RUN mkdir -p "/tmp/sndfile"

RUN yum install -y autoconf autogen automake build-essential libasound2-dev \
  libflac-dev libogg-dev libtool libvorbis-dev libopus-dev pkg-config gcc-c++

WORKDIR "/tmp/sndfile"

RUN curl -L -o "libsndfile-${SNDFILE_VERSION}.tar.gz" "http://www.mega-nerd.com/libsndfile/files/libsndfile-${SNDFILE_VERSION}.tar.gz"

RUN tar xf "libsndfile-${SNDFILE_VERSION}.tar.gz"

WORKDIR "/tmp/sndfile/libsndfile-${SNDFILE_VERSION}"

RUN ./configure --prefix=/opt/
RUN make
RUN make install

# set workdir back
WORKDIR /var/task

# Copy the earlier created requirements.txt file to the container
COPY requirements.txt ./

# Install the python requirements from requirements.txt
RUN python3.7 -m pip install -r requirements.txt

# Copy training model
COPY Early_CoughCovid_ResNet50_15_02_2022.hdf5 ./

# Copy a test file
COPY test.wav ./

# Ignore python warning
ENV PYTHONWARNINGS=ignore

# Copy the main app.py file to the container
COPY app.py ./


# Set the CMD to your handler
CMD ["app.lambda_handler"]
