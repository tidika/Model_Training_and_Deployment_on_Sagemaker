# SageMaker sklearn base image
FROM 257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3

RUN pip3 install matplotlib

ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker sklearn container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY train.py /opt/ml/code/train.py

# Defines train.py as script entrypoint 
ENV SAGEMAKER_PROGRAM train.py