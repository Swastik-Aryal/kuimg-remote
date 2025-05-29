###ALL IN ONE####
def dynFunc():
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
    from dynamic_inx.model_loader import my_loaded_model

    import gensim
    from gensim.models import KeyedVectors
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.manifold import TSNE
    import numpy as np
    from scipy.cluster.hierarchy import linkage, dendrogram
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    catalog_path = config.MODEL_CATALOG_PATH

    def parse_catalog_tags() : 
        model_info = {}
        tag_model_list = defaultdict(set)
        #added by me
        all_tags=[]

        with open(catalog_path, 'r') as f : 
            entries = json.loads(f.read())
        
            for entry in entries : 
                MODULE = entry['module']
                CLASS = entry['class']
                RAW = entry['raw']
                MODEL = getattr(import_module(MODULE), CLASS)
                BASE = inspect.getmro(MODEL)[-2]	
                TAGS = MODEL.tags
                
                all_tags.extend(TAGS)
        
                _model_info = {
                    'module' : MODULE,
                    'class' : CLASS,
                    'raw' : RAW,
                    'model' : MODEL,
                    'base' : BASE,
                    'tags' : TAGS
                }
        
                model_info[MODEL] = _model_info
        
                for tag in TAGS : 
                    tag_model_list[tag].add(MODEL)
                    
        return model_info, tag_model_list

        # return all_tags
        
        # return all_tags

    # build the list of all tags structures
    model_info, tag_model_list = parse_catalog_tags()

    #model = KeyedVectors.load_word2vec_format('app/dynamic_inx/PreTrainedW2V_model/GoogleNews-vectors-negative300.bin', binary=True)
    model = my_loaded_model
    print("model loaded")
    
    def hierarchical_clustering(words_for_tsne, model):
        word_vectors_for_clustering = np.array([model[word] for word in words_for_tsne])
        
    #     # Step 5: Visualize the hierarchical structure using tree diagrams
    #     linkage_matrix = None
    #     #firstway :problem doesnt show the cluser with linkage_matrix
    # #     #    # Perform hierarchical clustering using Ward's method
    #     try:
            
        linkage_matrix = linkage(word_vectors_for_clustering, method='ward')
            
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, labels=words_for_tsne, leaf_font_size=10)
        plt.xlabel('Words')
        plt.ylabel('Distance')
        plt.title('Hierarchical Clustering of Words')
        plt.show()
            
        # # #     #secondway by first computing cosine distances of vectors.
        # #     # Compute the cosine distances between word vectors
        # #     distance_matrix = cosine_distances(word_vectors_for_clustering)
        # #     linkage_matrix = linkage(distance_matrix, method='ward')
        # #     plt.figure(figsize=(12, 8))
        # #     dendrogram(linkage_matrix, labels=words_for_tsne, leaf_font_size=10)
        # #     plt.xlabel('Words')
        # #     plt.ylabel('Distance')
        # #     plt.title('Hierarchical Clustering of Words')
        # #     plt.show()
        # except ValueError as e:
        #     print("Error:",e)
        #     print("Hierarchical clustering couldn't be performed due to an empty distance matrix. Please provide two or more tags first.")
        return linkage_matrix

    def t_sne_visualization(words_for_tsne, model, perplexity_fraction=0.3):
        # Step 2: Get word embeddings for words used in t-SNE visualization
        word_vectors_for_tsne = np.array([model[word] for word in words_for_tsne])

        # Step 3: Calculate perplexity as a fraction of the number of words
        perplexity_fraction = 0.3  # Set your desired fraction here
        perplexity = int(len(words_for_tsne) * perplexity_fraction)

        # Step 4: Perform t-SNE to reduce the word embeddings to 2D
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(word_vectors_for_tsne)

        # Step 5: Calculate a single number to represent the vector
        def vector_summary(vector):
            return np.sum(vector)  # You can use any other summary metric if needed

        vector_summaries = [vector_summary(vec) for vec in word_vectors_for_tsne]

        #### uncomment below to plot in the terminal itself....
        # # Step 6: Create a scatter plot using Plotly Express with click events
        # fig = go.Figure()

        # for i, word in enumerate(words_for_tsne):
        #     x, y = embeddings_2d[i]
        #     vec_value = vector_summaries[i]
        #     fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker=dict(size=10),
        #                             text=[f'{word}<br>Vector Value: {vec_value:.2f}'], name=word))

        # fig.update_layout(title='Word Embedding Axes Diagram using t-SNE (2D)',
        #                 xaxis_title='t-SNE Dimension 1',
        #                 yaxis_title='t-SNE Dimension 2',
        #                 hovermode='closest')
        # fig.show()
        
        return embeddings_2d, vector_summaries



    def generate_word_cloud(text_list):
        # Count the frequency of each word in the list
        word_freq = {}
        for word in text_list:
            word_freq[word] = word_freq.get(word, 0) + 1
        # Create a WordCloud object using the word frequencies
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(word_freq)

        # Display the generated word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    # # ## uncomment this for test propose of manual list given ##
    # new_tag = 'BMW'
    # words_for_tsne = ['cat', 'airplane','car', 'dog','cycle', 'apple', 'chair', 'banana', 'orange', 'grape', 'mammal', 'fruits', 'plant', 'table', 'furniture', 'horse']
    # #checking if the new tage is already in the list or not.
    # if new_tag in words_for_tsne:
    #     x=1
    #     print(f"The given tag {new_tag} is already in the list")
    # else:
    #     x=0
    #     print(f"The given tag {new_tag} is not in the list")
    # # Adding the new tag in our old tags list.
    # words_for_tsne.append(new_tag)
    # ## uncomment section till here for the manual list given ##
    dummy=[]
    my_formatted_data = {str(k): v['tags'] for k, v in model_info.items()}
    lisst = [i for i in my_formatted_data.values()]
    flattened_list = [item for sublist in lisst for item in sublist if item != "other"]
    words_for_tsne = flattened_list
    words_for_tsne.extend(dummy)
    
    print("\n words_for_tsne also loaded", words_for_tsne)
    list_len = len(words_for_tsne)
    print("\n length of the list:", list_len )
    # my_formatted_data = {str(k): v['tags'] for k, v in mti.model_info.items()}
    # lisst = [i for i in my_formatted_data.values()]
    # flattened_list = [item for sublist in lisst for item in sublist if item != "other"]
    # words_for_tsne = flattened_list

    ## for working with the sailav's provided list from dynamically generated tags.
    # words_for_tsne=new_all_tags
    # Convert list to set for tasks 1 and 2 cause they requires non repeating tags (unlike list, set has uniques values)
    if list_len > 3:
        
        words_set_for_clustering = set(words_for_tsne) 
        words_set_for_clustering = list(words_set_for_clustering)
        whole_list_of_tags=words_set_for_clustering
        
        # Task 1: Hierarchical Clustering
        linkage_matrix = hierarchical_clustering(words_set_for_clustering, model)

        # Task 2: t-SNE Visualization
        embeddings_2d, vector_summaries = t_sne_visualization(words_set_for_clustering, model)

        # Task 3: Word Cloud Generation
        generate_word_cloud(words_for_tsne)

        # Get cluster labels based on a threshold (you can adjust this threshold based on the dendrogram)
        threshold = 4.8  # Adjust this threshold as needed '2' is good for 2nd clutering method.
        cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')

        #Building the Pandas DataFrame with columns having the name of tags, embeddings and other information collected from above functions.
        # Collect data for the DataFrame
        data = []
        for i, word in enumerate(whole_list_of_tags):
            x, y = embeddings_2d[i]
            vector_value = vector_summaries[i]
            cluster_name = f"Category_{cluster_labels[i]}"
            data.append((word, cluster_name, x, y, vector_value))

        # Create pandas DataFrame
        columns = ['Tag', 'Cluster_Name', 't-SNE_Dimension_1', 't-SNE_Dimension_2', 'Vector_Value']
        df = pd.DataFrame(data, columns=columns)

        # Print the DataFrame
        print(df)
        data_table=df
        # if we have the DataFrame df
        sorted_df = df.sort_values(by='Cluster_Name')
        #     sorted_df = df.sort_values(by='Cluster_Name', ascending=False)

        # Print the sorted DataFrame
        print(sorted_df)


        ## Pandas manupulations:

        # 1. Get the list of tags
        tags_list = df['Tag'].tolist()
        print("List of Tags:", tags_list)

            # Create a dictionary of similar categories
        similar_categories_dict = {}
        for cluster_name in df['Cluster_Name'].unique():
            cluster_tags = df[df['Cluster_Name'] == cluster_name]['Tag'].tolist()
            similar_categories_dict[cluster_name] = cluster_tags

        print("\nDictionary of Similar Categories:")
        print(similar_categories_dict)

        # Convert similar_categories_dict to JSON
        similar_categories_dict_json = json.dumps(similar_categories_dict, indent=4)
        print("\nDictionary of Similar Categories (JSON format):")
        print(similar_categories_dict_json)
        return data_table

        
        #     # Print the 2D embeddings
        #     print("t-SNE 2D Embeddings:")
        #     for i, (x, y) in enumerate(embeddings_2d):
        #         print(f"Tag: {whole_list_of_tags[i]}, Embedding: ({x:.2f}, {y:.2f}), Vector Value: {vector_summaries[i]:.2f},  Hierarchy Order: {i}")


        #     # Export the DataFrame to a database
        # # First u should have database and a table called "tags_table"
        # your_database_conn = your_database.connect()  # Replace 'your_database' with the actual database connection
        # df.to_sql('your_table', your_database_conn, if_exists='replace', index=False)

    else:
        print("Hierarchical clustering couldn't be performed due to an empty distance matrix, or 'perplexity' parameter of TSNE must be a float (0.0), but got 0. Please provide two or more words for clustering.")
        # return "None"
    
