###ALL IN ONE####

import config
import json

# DYNAMIC IMPORTS
from importlib import import_module

# DYNAMIC INSPECTS
import inspect

# DEFAULT DICT
from collections import defaultdict
from pprint import pprint

# BASE MODELS
from models.base import MultiClassSingleTagModel

import json
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cosine
import models.tag_interface as mti

# from model_loader import loaded_model
# from hierarchical_clustering import hierarchical_clustering
# from t_sne_visualization import t_sne_visualization
# from word_cloud import generate_word_cloud
# from dynamic_inx.model_loader import loaded_model
# from dynamic_inx.hierarchical_clustering import hierarchical_clustering
# from dynamic_inx.t_sne_visualization import t_sne_visualization
# from dynamic_inx.word_cloud import generate_word_cloud

import gensim
from gensim.models import KeyedVectors
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from wordcloud import WordCloud
import matplotlib.pyplot as plt


#model = KeyedVectors.load_word2vec_format('app/dynamic_inx/PreTrainedW2V_model/GoogleNews-vectors-negative300.bin', binary=True)
model = KeyedVectors.load_word2vec_format('dynamic_inx/PreTrainedW2V_model/GoogleNews-vectors-negative300.bin', binary=True)
my_loaded_model = model