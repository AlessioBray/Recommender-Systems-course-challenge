"""
Created on 11/12/21
@author: Alessio Bray
"""

from Recommenders.Recommender_utils import check_matrix
import scipy.sparse as sps

import numpy as np
from numpy import linalg as LA

from Recommenders.BaseRecommender import BaseRecommender



class DifferentLossScoresHybridRecommender(BaseRecommender):
    """ ScoresHybridRecommender
    Hybrid of two prediction scores R = R1/norm*alpha + R2/norm*(1-alpha) where R1 and R2 come from
    algorithms trained on different loss functions.

    """

    RECOMMENDER_NAME = "DifferentLossScoresHybridRecommender"


    def __init__(self, URM_train, recommenders):
        super(DifferentLossScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommenders[0]
        self.recommender_2 = recommenders[1]
        
        
        
    def fit(self, norm, alpha = 0.5):

        self.alpha = alpha
        self.norm = norm


    def _compute_item_score(self, user_id_array, items_to_compute = None):
        
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm)
        
        
        if norm_item_weights_1 == 0:
            raise ValueError("Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))
        
        if norm_item_weights_2 == 0:
            raise ValueError("Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))
        
        item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * (1-self.alpha)

        return item_weights

    
class N_DifferentLossScoresHybridRecommender(BaseRecommender):
    """ ScoresHybridRecommender
    Hybrid of two prediction scores R = R1/norm*alpha + R2/norm*(1-alpha) where R1 and R2 come from
    algorithms trained on different loss functions.

    """

    RECOMMENDER_NAME = "N_DifferentLossScoresHybridRecommender"


    def __init__(self, URM_train, recommenders, number_of_recommenders):
        super(N_DifferentLossScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_array = recommenders
        self.number_of_recommenders = number_of_recommenders
        
        
    def fit(self, norm, alpha = 0, beta = 0, gamma = 0, delta = 0):

        self.weight_array = [alpha, beta, gamma, delta]
        self.norm = norm


    def _compute_item_score(self, user_id_array, items_to_compute):
        
        item_weights = []
        norm_item_weights = []
        for i in range(self.number_of_recommenders):
            
            item_weights.append(self.recommender_array[i]._compute_item_score(user_id_array))
            norm_item_weights.append(LA.norm(item_weights[i], self.norm))
        
        
            if norm_item_weights[i] == 0:
                raise ValueError("Norm {} of item weights for recommender {} is zero. Avoiding division by zero".format(self.norm, i))
        
        weighted_matrices = [a * b / c for a, b, c in zip(item_weights, self.weight_array[:len(item_weights)], norm_item_weights)]
        
        item_weights = sum(weighted_matrices)

        return item_weights        