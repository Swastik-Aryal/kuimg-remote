import logging 
import os

# ENVIRONMENT VARIABLES AND PATHS ----------------------------------------------
# Specify environment and file paths and ensure that they exist
# ------------------------------------------------------------------------------
APP_NAME = os.environ['APP_NAME']
APP_PATH = os.environ['APP_PATH']
TEMP_PATH = os.path.join(APP_PATH, 'temp')
TEMP_DATA_PATH = os.path.join(TEMP_PATH, 'data')
TEMP_RAW_PATH = os.path.join(TEMP_PATH, 'raw')
if not os.path.exists(TEMP_PATH) : os.makedirs(TEMP_PATH)
if not os.path.exists(TEMP_DATA_PATH) : os.makedirs(TEMP_DATA_PATH)
if not os.path.exists(TEMP_RAW_PATH) : os.makedirs(TEMP_RAW_PATH)
# ------------------------------------------------------------------------------

# MODEL CONFIGURATION
# ------------------------------------------------------------------------------
MODEL_PATH = os.path.join(APP_PATH, 'models')
MODEL_RAW_PATH = os.path.join(MODEL_PATH, 'raw')
MODEL_STANDARD_PATH = os.path.join(MODEL_PATH, 'standard')
MODEL_CATALOG_PATH = os.path.join(MODEL_PATH, 'catalog.json')
MODEL_TEMPLATE_PATH = os.path.join(MODEL_PATH, 'templates')
if not os.path.exists(MODEL_PATH) : os.makedirs(MODEL_PATH)
if not os.path.exists(MODEL_RAW_PATH) : os.makedirs(MODEL_RAW_PATH)
if not os.path.exists(MODEL_TEMPLATE_PATH) : os.makedirs(MODEL_TEMPLATE_PATH)
# ------------------------------------------------------------------------------
# AUTOTAG CONFIGURATION 
# ------------------------------------------------------------------------------
# AUTO_IMG_DFLT_KWRD_MAX = 50
# AUTO_IMG_DFLT_CNTR_MAX = 10
AUTO_IMG_DFLT_KWRD_MAX = 64
AUTO_IMG_DFLT_CNTR_MAX = 16
TEMP_AUTO_IMG_PATH = os.path.join(TEMP_PATH, 'auto', 'images')
TEMP_AUTO_RAW_PATH = os.path.join(TEMP_PATH, 'auto', 'raw')
if not os.path.exists(TEMP_AUTO_IMG_PATH) : os.makedirs(TEMP_AUTO_IMG_PATH)
if not os.path.exists(TEMP_AUTO_RAW_PATH) : os.makedirs(TEMP_AUTO_RAW_PATH)
# ------------------------------------------------------------------------------
#GAN CONFIG
GAN_BATCH_SIZE=32
LATENT_VEC_DIMENSION=100
IMAGE_TO_GENERATE=100
# ------------------------------------------------------------------------------
# STORAGE SERVER CONFIG
# ------------------------------------------------------------------------------
REDIS_FS = os.environ['REDIS_FS']
# ------------------------------------------------------------------------------

