# DOCKER BUILD AND RUN
`docker build --network=host -t kuimg .`

`docker run -it -p 8000:80 kuimg`

# DOCKER-COMPOSE DEPLOYMENT
`docker-compose --compatibility up --build`

# For Running locally

## Start redis
There is a lot of difficulty to run redis locally, so spin up a redis container and run it in port 6379

## Linux

### Environment variables
`cd <the folder containing kuimg>`

`source ./local_env.sh`

### Start the application
`cd app`

`uvicorn main:app --host 0.0.0.0 --port 80 --log-level debug`

If the port 80 isn't available or cannot be binded due to permission issue use other ports

## Windows


### Environment variables
`cd <the folder containing kuimg>`

`./local_env.ps1`

### Start the application
`cd app`

`uvicorn main:app --host 0.0.0.0 --port 80 --log-level debug`