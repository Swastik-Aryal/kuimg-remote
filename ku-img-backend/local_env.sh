#!/bin/sh

export APP_NAME=kuimg
echo "Exported APP_NAME=$APP_NAME."

working_dir="$(pwd)"
echo "The current working directory is $working_dir."

export APP_PATH="$working_dir/app/"
echo "Exported APP_PATH=$APP_PATH."

export REDIS_FS="localhost:6379"
echo "Exported REDIS_FS=$REDIS_FS."
