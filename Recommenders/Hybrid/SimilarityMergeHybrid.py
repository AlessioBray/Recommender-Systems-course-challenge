#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/12/21

@author: Alessio Bray
"""

from Recommenders.Recommender_import_list import *

from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender


class SimilarityMergeHybridRecommender(BaseItemSimilarityMatrixRecommender):
    """SimilarityMergeHybridRecommender"""

    RECOMMENDER_NAME = "SimilarityMergeHybridRecommender"

    def __init__(self, URM_train, recommenders, verbose=True):
        super(SimilarityMergeHybridRecommender, self).__init__(URM_train, verbose = verbose)

        self._URM_train_format_checked = False
        self._W_sparse_format_checked = False

        self.recommenders = recommenders

    def fit(self, alpha, selectTopK = False, topK=100):

        W_sparse = (1 - alpha) * self.recommenders[0].W_sparse + alpha * self.recommenders[1].W_sparse

        assert W_sparse.shape[0] == W_sparse.shape[1],\
            "ItemKNNCustomSimilarityRecommender: W_sparse matrice is not square. Current shape is {}".format(W_sparse.shape)

        assert self.URM_train.shape[1] == W_sparse.shape[0],\
            "ItemKNNCustomSimilarityRecommender: URM_train and W_sparse matrices are not consistent. " \
            "The number of columns in URM_train must be equal to the rows in W_sparse. " \
            "Current shapes are: URM_train {}, W_sparse {}".format(self.URM_train.shape, W_sparse.shape)

        if selectTopK:
            W_sparse = similarityMatrixTopK(W_sparse, k=topK)

        self.W_sparse = check_matrix(W_sparse, format='csr')