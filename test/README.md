## Batch job

Batch prediction job may be executed using docker image `my_docker_image` created earlier (see [deployment_docker/README.md](../deployment_docker/README.md)).  
You will have to mount the folder where input data is located (output will also go here) and pass input and output file names as parameters to docker, e.g:  
`docker run --rm -ti -v {data_folder}:/app/data/ my_docker_image ./data/{input_file} ./data/{output_file}` 

In my case the actuall command was the following:  
`docker run --rm -ti -v c:/_temp/data/:/app/data/ my_docker_image ./data/input.csv ./data/output.csv`
