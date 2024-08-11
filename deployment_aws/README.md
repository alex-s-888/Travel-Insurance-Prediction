## Deploy model to AWS SageMaker

Assuming you have active AWS account, the following steps are needed:

Install the AWS CLI (e.g. using `pip install awscli`).  
Configure AWS credentials.  
Set up proper AWS IAM role, see [Amazon SageMaker Prerequisites](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html).  
Execute command (it creates the docker image with utilities used by MLflow, registers in AWS ECR)  
`mlflow sagemaker build-and-push-container`

Now the actual model deployment is done by the following command
```
mlflow sagemaker deploy --app-name <your-app-name> \
    --model-uri "models:/<model-name>/<model-version>" \
    -e <iam-role> \
    --region-name <region>
```


### Reference
[DagsHub Docs  - Deploy ML Models to a Cloud Platform](https://dagshub.com/docs/use_cases/deploy_ml_model_to_cloud/)
