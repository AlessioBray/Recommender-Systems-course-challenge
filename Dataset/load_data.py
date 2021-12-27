"""
Created on 06/12/21

@author: Alessio Bray
"""

import numpy as np
import pandas as pd
import scipy.sparse as sps
import os


def preprocess_ICM(ICM_name, ICM):

    if ICM_name == 'ICM_event':
        df = ICM.groupby(['ItemID']).count()
        df = df.reset_index()
        df.Feature = 1
    
    return df
    


def load_data():

    DATA_FILE_PATH = "./Dataset"

    URM_PATH = os.path.join(DATA_FILE_PATH, "data_train.csv")

    ICM_channel_PATH = os.path.join(DATA_FILE_PATH, "data_ICM_channel.csv")
    ICM_event_PATH = os.path.join(DATA_FILE_PATH, "data_ICM_event.csv")
    ICM_genre_PATH = os.path.join(DATA_FILE_PATH, "data_ICM_genre.csv")
    ICM_subgenre_PATH = os.path.join(DATA_FILE_PATH, "data_ICM_subgenre.csv")

    ## URM

    URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_PATH, 
                                sep=",",
                                dtype={0:int, 1:int, 2:int},
                                engine='python')

    URM_all_dataframe.columns = ["UserID", "ItemID", "Interaction"]

    ## ICMs

    ICM_genre_dataframe = pd.read_csv(filepath_or_buffer=ICM_genre_PATH, 
                            sep=",", 
                            dtype={0:int, 1:int, 2:int},
                            engine='python')

    ICM_genre_dataframe.columns = ["ItemID", "GenreID", "Feature"]

    ICM_subgenre_dataframe = pd.read_csv(filepath_or_buffer=ICM_subgenre_PATH, 
                            sep=",", 
                            dtype={0:int, 1:int, 2:int},
                            engine='python')

    ICM_subgenre_dataframe.columns = ["ItemID", "SubgenreID", "Feature"]

    ICM_event_dataframe = pd.read_csv(filepath_or_buffer=ICM_event_PATH, 
                            sep=",", 
                            dtype={0:int, 1:int, 2:int},
                            engine='python')

    ICM_event_dataframe.columns = ["ItemID", "EventID", "Feature"]

    ICM_channel_dataframe = pd.read_csv(filepath_or_buffer=ICM_channel_PATH, 
                            sep=",", 
                            dtype={0:int, 1:int, 2:int},
                            engine='python')

    ICM_channel_dataframe.columns = ["ItemID", "ChannelID", "Feature"]

    ## Preprocess ICMs

    #ICM_event = preprocess_ICM('ICM_event', ICM_event_dataframe)

    ## Data Analysis

    userID_unique = URM_all_dataframe["UserID"].unique()
    itemID_unique = URM_all_dataframe["ItemID"].unique()

    n_users = len(userID_unique)
    n_items = len(itemID_unique)

    n_genre_features = len(ICM_genre_dataframe["GenreID"].unique())
    n_subgenre_features = len(ICM_subgenre_dataframe["SubgenreID"].unique())
    n_event_features = max(ICM_event_dataframe["EventID"]) + 1 # this must be set to max since it is a processed matrix
    n_channel_features = len(ICM_channel_dataframe["ChannelID"].unique())

    ## Turning into sparse matrices

    URM_all = sps.csr_matrix((URM_all_dataframe["Interaction"].values, (URM_all_dataframe["UserID"].values, URM_all_dataframe["ItemID"].values)),
                        shape = (n_users, n_items))

    ICM_genre = sps.csr_matrix((np.ones(len(ICM_genre_dataframe["ItemID"].values)), (ICM_genre_dataframe["ItemID"].values, ICM_genre_dataframe["GenreID"].values)),
                           shape = (n_items, n_genre_features)
                          )

    ICM_genre.data = np.ones_like(ICM_genre.data)

    ICM_subgenre = sps.csr_matrix((np.ones(len(ICM_subgenre_dataframe["ItemID"].values)), (ICM_subgenre_dataframe["ItemID"].values, ICM_subgenre_dataframe["SubgenreID"].values)),
                               shape = (n_items, n_subgenre_features)
                             )

    ICM_subgenre.data = np.ones_like(ICM_subgenre.data)

    ICM_event = sps.csr_matrix((np.ones(len(ICM_event_dataframe["ItemID"].values)), (ICM_event_dataframe["ItemID"].values, ICM_event_dataframe["EventID"].values)),
                            shape = (n_items, n_event_features)
                             )

    ICM_event.data = np.ones_like(ICM_event.data)

    ICM_channel = sps.csr_matrix((np.ones(len(ICM_channel_dataframe["ItemID"].values)), 
                           (ICM_channel_dataframe["ItemID"].values, ICM_channel_dataframe["ChannelID"].values)),
                           shape = (n_items, n_channel_features)
                          )

    ICM_channel.data = np.ones_like(ICM_channel.data)

    return URM_all, {'ICM_genre': ICM_genre, 'ICM_subgenre': ICM_subgenre, 'ICM_event': ICM_event, 'ICM_channel': ICM_channel}