"""
Created on 06/11/21
@author: Alessio Bray
"""

import os
import pandas as pd
import numpy as np

def load_test_user_array():

    DATA_FILE_PATH = './Dataset'

    test_UserID_path = os.path.join(DATA_FILE_PATH, "data_target_users_test.csv")

    test_UserID_dataframe = pd.read_csv(filepath_or_buffer=test_UserID_path, 
                                        sep=",", 
                                        dtype={0:int},
                                        engine='python')

    test_UserID_dataframe.columns = ["UserID"]

    return test_UserID_dataframe.to_numpy().ravel().tolist()