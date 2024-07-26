# Model_Training_and_Deployment_on_Sagemaker

This repository contains code to train and deploy a tensorflow model using aws sagemaker. 

[data_processing.ipynb](/data_processing.ipynb) file is used to process and store training and test data in an s3 bucket.  

[train.py](/train.py) file is used to write code used for training the model. 

[inference_script.py](/inference_script.py) file is used to write script that instructs inference container on where to fetch the model for inference. 


[train_and_deploy_job.ipynb](/inference_script.py) file is used to orchestrate the training and deployment of the model. 
