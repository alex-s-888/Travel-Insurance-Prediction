# Travel Insurance Prediction

The goal of the project is to build and deploy Machine Learning model that predicts if customer will buy travel insurance policy based on customer's profile. 

Imagine following use case: company is planning promo campaign where it will send some gifts to encourage customers buy policies. 
The budget of campaign is limited so gifts should be sent only to customers that are likely to do the purchase.


## Prerequisites

The project uses [DagsHub](https://dagshub.com/) as Cloud platform. It provides such services as MLflow server, model registry, data storage and may be used for model deployment.  
Valid DagsHub account is required for this project (free Community option available). 

This is python (ver 3.11) project. The following packages should be present:
- dagshub
- pandas
- numpy
- scikit-learn
- mlflow
- joblib
- pylint
- pre-commit

(Optional) Docker should be installed if you want to deploy resulting model as docker image.    
(Optional) AWS account is needed if you want to deploy resulting model to AWS SageMaker.  


## Project structure

`.pre-commit-config.yaml` - pre-commit hooks configuration.

### dataset
Folder contains [TravelInsurancePrediction.csv](dataset/TravelInsurancePrediction.csv) spreadsheet with customers' data.

### model
Contains the source code  
- [experiments.py](model/experiments.py) used for experiments tracking with MLflow.
- [build.py](model/build.py) builds model based on experiments metrics and serializes it as `.joblib` file.

### deployment_docker
Folder contains files necessary to deploy model as Docker image.

### deployment_aws
Folder contains instructions how to deploy model to AWS SageMaker.

### test
Instructions how to use/test predictions using dockerized model. 


## Walk-through

First, you should authorize DagsHub client, see [DagsHub Client Docs](https://dagshub.com/docs/client/reference/auth.html).    
Imho, easiest way is by doing `dagshub login` in CLI.

Next step is to execute `model/experiments.py` code. It will do the following:
- read dataset and apply necessary data transformations, clean-up, etc...
- split data to Train and Test sets
- do several runs with different hyperparameters using `GradientBoostingClassifier`
- track experiments using MLflow server
- choose best hyperparameters values
- register model/version and mark state as `Staging`  

(Optional, needed for model deployment as docker image) Execute `model/build.py` code:
- build model based on best hyperparameters values
- serialize model using `joblib` format, resulting file `my_model.joblib` will be located in `deployment_docker` folder  
After the model is built, it may be deployed as Docker image. See instructions [deployment_docker/README.md](deployment_docker/README.md) how to use `Dockefile` and other stuff.  
The dockerized model may be used to execute batch prediction jobs, see example [test/README.md](test/README.md).

&nbsp;  
Screenshot examples of DagsHub/MLflow UI:  

![mlflow_001.png](screenshots%2Fmlflow_001.png)   

&nbsp;    

![mlflow_002.png](screenshots%2Fmlflow_002.png)
