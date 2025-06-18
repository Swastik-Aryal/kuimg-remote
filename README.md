# kuimg-remote
This is the server version of the ku-img app. 

P.S. This will not run locally.
## Steps

1. Frontend
    - `cd ku-img-frontend`
    - `npm install`
    - `npm run build`

2. Backend
    - `cd ku-img-backend`
    - `sudo chown -R $USER:$USER /home/ubuntu/kuimg-remote/`
    - `docker-compose up --build`
