import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from model_loader import loaded_model

def hierarchical_clustering(words_for_tsne, model):
    word_vectors_for_clustering = np.array([model[word] for word in words_for_tsne])
    linkage_matrix = linkage(word_vectors_for_clustering, method='ward')

    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=words_for_tsne, leaf_font_size=10)
    plt.xlabel('Words')
    plt.ylabel('Distance')
    plt.title('Hierarchical Clustering of Words')
    plt.show()
    
    return linkage_matrix
