#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/12/21

@author: Alessio Bray
"""

import scipy.sparse as sps
import numpy as np

from Recommenders.Recommender_import_list import *
from Recommenders.BaseRecommender import BaseRecommender

class P3alphaICMweightedHybrid(BaseRecommender):
    """ICMweightedHybrid"""

    RECOMMENDER_NAME = "ICMweightedHybrid"

    def __init__(self, URM_train, ICMs, recommender_params):
        super(ICMweightedHybrid, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.ICMs= ICMs
        self.recommender_params = recommender_params

    def fit(self, alpha = 0.5, beta = 0.5, gamma = 0, delta = 0.5):

        self.weights = [alpha, beta, gamma, delta]
        stacked_URM = self.URM_train

        for i in range(len(self.ICMs)):
            self.ICMs[i] = self.ICMs[i] * self.weights[i]
            stacked_URM =  sps.vstack([stacked_URM.tocoo(), self.ICMs[i].T.tocoo()], format='csr')
        
        self.recommender = P3alphaRecommender(stacked_URM)
        self.recommender.fit(**self.recommender_params)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        
        item_weights = self.recommender._compute_item_score(user_id_array)

        return item_weights




class SLIMElasticNetICMweightedHybrid(BaseRecommender):
    """ICMweightedHybrid"""

    RECOMMENDER_NAME = "ICMweightedHybrid"

    def __init__(self, URM_train, ICMs, recommender_params):
        super(SLIMElasticNetICMweightedHybrid, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.ICMs = ICMs
        self.recommender_params = recommender_params

    def fit(self, alpha = 0.5, beta = 0.5, gamma = 0, delta = 0.5):

        self.weights = [alpha, beta, gamma, delta]
        stacked_URM = self.URM_train

        for i in range(len(self.ICMs)):
            self.ICMs[i] = self.ICMs[i] * self.weights[i]
            stacked_URM =  sps.vstack([stacked_URM.tocoo(), self.ICMs[i].T.tocoo()], format='csr')
        
        self.recommender = P3alphaRecommender(stacked_URM)
        self.recommender.fit(**self.recommender_params)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        
        item_weights = self.recommender._compute_item_score(user_id_array)

        return item_weights