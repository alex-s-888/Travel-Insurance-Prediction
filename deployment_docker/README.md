## Build docker image

After the model is built (file `my_model.joblib` is placed in `deployment_docker`), from `deployment_docker` folder execute:  
`docker build --tag my_docker_image .`