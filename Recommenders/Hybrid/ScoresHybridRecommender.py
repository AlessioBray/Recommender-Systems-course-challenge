"""
Created on 06/12/21
@author: Alessio Bray
"""

from Recommenders.Recommender_utils import check_matrix
import scipy.sparse as sps

import numpy as np



from Recommenders.BaseRecommender import BaseRecommender

class ScoresHybridRecommender(BaseRecommender):

    """ ScoresHybridRecommender
    Hybrid of 2 prediction scores R
    """

    RECOMMENDER_NAME = "ScoresHybridRecommender"

    def __init__(self, URM_train, recommenders):
        super(ScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommenders[0]
        self.recommender_2 = recommenders[1]

    def fit(self, alpha = 0.5):
        self.alpha = alpha
        

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        
        # In a simple extension this could be a loop over a list of pretrained recommender objects
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        item_weights = item_weights_1*self.alpha + item_weights_2*(1-self.alpha)

        return item_weights
    

class NScoresHybridRecommender(BaseRecommender):

    """ NScoresHybridRecommender
    Hybrid of N prediction scores R, with N from 1 to 4
    """

    RECOMMENDER_NAME = "ScoresHybridRecommender"

    def __init__(self, URM_train, recommender_array, number_of_recommenders):
        super(ScoresHybridRecommender, self).__init__(URM_train)

        self.number_of_recommenders = number_of_recommenders
        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.recommender_array = recommender_array
        print('Number of recommenders: ', self.number_of_recommenders)

    def fit(self, alpha, beta, gamma, delta):
        self.weight_array = [alpha, beta, gamma, delta]

    def _compute_item_score(self, user_id_array, items_to_compute):

        item_weights = []
        for i in range(self.number_of_recommenders):
            
            item_weights.append(self.recommender_array[i]._compute_item_score(user_id_array))

        weighted_matrices = [a * b for a, b in zip(item_weights, self.weight_array[:len(item_weights)])]
        #print(weighted_matrices[0].shape)
        # item_weights_1 = self.Recommender_1._compute_item_score(user_id_array)
        # item_weights_2 = self.Recommender_2._compute_item_score(user_id_array)
        # item_weights_3 = self.Recommender_3._compute_item_score(user_id_array)
        # print('shape of item_weights hybrid N:', item_weights_1.shape)

        # item_weights = np.multiply(item_weights, self.weight_array)
        # item_weights = item_weights_1*self.alpha + item_weights_2*self.beta + item_weights_3*(1-self.alpha-self.beta)
        item_weights = sum(weighted_matrices)

        return item_weights