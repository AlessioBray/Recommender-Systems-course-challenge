"""
Created on 07/12/21

@author: Alessio Bray
"""

import os, multiprocessing
from functools import partial


######################################################################
##########                                                  ##########
##########                  PURE COLLABORATIVE              ##########
##########                                                  ##########
######################################################################
from Recommenders.NonPersonalizedRecommender import TopPop, Random, GlobalEffects

# KNN
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

# KNN machine learning
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender

# Matrix Factorization
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender, PureSVDItemRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython,\
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython

from Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask as MultVAERecommender
from Recommenders.FactorizationMachines.LightFMRecommender import LightFMCFRecommender

######################################################################
##########                                                  ##########
##########                  PURE CONTENT BASED              ##########
##########                                                  ##########
######################################################################
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender



######################################################################
##########                                                  ##########
##########                       HYBRID                     ##########
##########                                                  ##########
######################################################################
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
from Recommenders.FactorizationMachines.LightFMRecommender import LightFMItemHybridRecommender, LightFMUserHybridRecommender
from Recommenders.FeatureWeighting.Cython.CFW_D_Similarity_Cython import CFW_D_Similarity_Cython
from Recommenders.FeatureWeighting.Cython.CFW_DVV_Similarity_Cython import CFW_DVV_Similarity_Cython
from Recommenders.FeatureWeighting.Cython.FBSM_Rating_Cython import FBSM_Rating_Cython

from Recommenders.Hybrid.ScoresHybridRecommender import ScoresHybridRecommender


######################################################################
from skopt.space import Real, Integer, Categorical
import traceback

from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

######################################################################

def runHyperparameterSearch_ScoresHybrid(recommenders, URM_train, ICM_object, ICM_name, URM_train_last_test = None,
                                        n_cases = None, n_random_starts = None, resume_from_saved = False,
                                        save_model = "no", evaluate_on_test = "no", max_total_time = None, evaluator_validation_earlystopping = None,
                                        evaluator_validation= None, evaluator_test=None, metric_to_optimize = None, cutoff_to_optimize = None,
                                        output_folder_path ="result_experiments/", parallelizeKNN = False, allow_weighting = True,
                                        similarity_type_list = None, user_id_array=None):
    """
    This function performs the hyperparameter optimization for a hybrid collaborative and content-based recommender

    :param recommenders:   List of classes of the recommender object to optimize, it must be a BaseRecommender type
    :param URM_train:           Sparse matrix containing the URM training data
    :param ICM_object:          Sparse matrix containing the ICM training data
    :param ICM_name:            String containing the name of the ICM, will be used for the name of the output files
    :param URM_train_last_test: Sparse matrix containing the union of URM training and validation data to be used in the last evaluation
    :param n_cases:             Number of hyperparameter sets to explore
    :param n_random_starts:     Number of the initial random hyperparameter values to explore, usually set at 30% of n_cases
    :param resume_from_saved:   Boolean value, if True the optimization is resumed from the saved files, if False a new one is done
    :param save_model:          ["no", "best", "last"] which of the models to save, see HyperparameterTuning/SearchAbstractClass for details
    :param evaluate_on_test:    ["all", "best", "last", "no"] when to evaluate the model on the test data, see HyperparameterTuning/SearchAbstractClass for details
    :param max_total_time:    [None or int] if set stops the hyperparameter optimization when the time in seconds for training and validation exceeds the threshold
    :param evaluator_validation:    Evaluator object to be used for the validation of each hyperparameter set
    :param evaluator_validation_earlystopping:   Evaluator object to be used for the earlystopping of ML algorithms, can be the same of evaluator_validation
    :param evaluator_test:          Evaluator object to be used for the test results, the output will only be saved but not used
    :param metric_to_optimize:  String with the name of the metric to be optimized as contained in the output of the evaluator objects
    :param cutoff_to_optimize:  Integer with the recommendation list length to be optimized as contained in the output of the evaluator objects
    :param output_folder_path:  Folder in which to save the output files
    :param parallelizeKNN:      Boolean value, if True the various heuristics of the KNNs will be computed in parallel, if False sequentially
    :param allow_weighting:     Boolean value, if True it enables the use of TF-IDF and BM25 to weight features, users and items in KNNs
    :param similarity_type_list: List of strings with the similarity heuristics to be used for the KNNs
    """

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    URM_train = URM_train.copy()
    #ICM_object = ICM_object.copy()

    if URM_train_last_test is not None:
        URM_train_last_test = URM_train_last_test.copy()


    earlystopping_keywargs = {"validation_every_n": 5,
                              "stop_on_validation": True,
                              "evaluator_object": evaluator_validation_earlystopping,
                              "lower_validations_allowed": 5,
                              "validation_metric": metric_to_optimize,
                              }

   ##########################################################################################################

    try:

        output_file_name_root = recommenders[0].RECOMMENDER_NAME + "_" + recommenders[1].RECOMMENDER_NAME

        hyperparameterSearch = SearchBayesianSkopt(ScoresHybridRecommender, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

        hyperparameters_range_dictionary = {}

        hyperparameters_range_dictionary["alpha"] = Real(low = 0.1, high = 0.9, prior = 'uniform')

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, recommenders],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {}
        )


        if URM_train_last_test is not None:
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
        else:
            recommender_input_args_last_test = None


        hyperparameterSearch.search(recommender_input_args,
                                    hyperparameter_search_space = hyperparameters_range_dictionary,
                                    n_cases = n_cases,
                                    n_random_starts = n_random_starts,
                                    resume_from_saved = resume_from_saved,
                                    save_model = save_model,
                                    evaluate_on_test = evaluate_on_test,
                                    max_total_time = max_total_time,
                                    output_folder_path = output_folder_path,
                                    output_file_name_root = output_file_name_root,
                                    metric_to_optimize = metric_to_optimize,
                                    cutoff_to_optimize = cutoff_to_optimize,
                                    recommender_input_args_last_test = recommender_input_args_last_test)

        



    except Exception as e:

        print("On recommender {} Exception {}".format(ScoresHybridRecommender.RECOMMENDER_NAME, str(e)))
        traceback.print_exc()

        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(ScoresHybridRecommender.RECOMMENDER_NAME, str(e)))
        error_file.close()

def runHyperparameterSearch_NScoresHybrid(recommenders, URM_train, ICM_object, ICM_name, URM_train_last_test = None,
                                        n_cases = None, n_random_starts = None, resume_from_saved = False,
                                        save_model = "no", evaluate_on_test = "no", max_total_time = None, evaluator_validation_earlystopping = None,
                                        evaluator_validation= None, evaluator_test=None, metric_to_optimize = None, cutoff_to_optimize = None,
                                        output_folder_path ="result_experiments/", parallelizeKNN = False, allow_weighting = True,
                                        similarity_type_list = None, user_id_array=None):
    """
    This function performs the hyperparameter optimization for a hybrid collaborative and content-based recommender

    :param recommenders:   List of classes of the recommender object to optimize, it must be a BaseRecommender type
    :param URM_train:           Sparse matrix containing the URM training data
    :param ICM_object:          Sparse matrix containing the ICM training data
    :param ICM_name:            String containing the name of the ICM, will be used for the name of the output files
    :param URM_train_last_test: Sparse matrix containing the union of URM training and validation data to be used in the last evaluation
    :param n_cases:             Number of hyperparameter sets to explore
    :param n_random_starts:     Number of the initial random hyperparameter values to explore, usually set at 30% of n_cases
    :param resume_from_saved:   Boolean value, if True the optimization is resumed from the saved files, if False a new one is done
    :param save_model:          ["no", "best", "last"] which of the models to save, see HyperparameterTuning/SearchAbstractClass for details
    :param evaluate_on_test:    ["all", "best", "last", "no"] when to evaluate the model on the test data, see HyperparameterTuning/SearchAbstractClass for details
    :param max_total_time:    [None or int] if set stops the hyperparameter optimization when the time in seconds for training and validation exceeds the threshold
    :param evaluator_validation:    Evaluator object to be used for the validation of each hyperparameter set
    :param evaluator_validation_earlystopping:   Evaluator object to be used for the earlystopping of ML algorithms, can be the same of evaluator_validation
    :param evaluator_test:          Evaluator object to be used for the test results, the output will only be saved but not used
    :param metric_to_optimize:  String with the name of the metric to be optimized as contained in the output of the evaluator objects
    :param cutoff_to_optimize:  Integer with the recommendation list length to be optimized as contained in the output of the evaluator objects
    :param output_folder_path:  Folder in which to save the output files
    :param parallelizeKNN:      Boolean value, if True the various heuristics of the KNNs will be computed in parallel, if False sequentially
    :param allow_weighting:     Boolean value, if True it enables the use of TF-IDF and BM25 to weight features, users and items in KNNs
    :param similarity_type_list: List of strings with the similarity heuristics to be used for the KNNs
    """

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    URM_train = URM_train.copy()
    #ICM_object = ICM_object.copy()

    if URM_train_last_test is not None:
        URM_train_last_test = URM_train_last_test.copy()


    earlystopping_keywargs = {"validation_every_n": 25,
                              "stop_on_validation": True,
                              "evaluator_object": evaluator_validation_earlystopping,
                              "lower_validations_allowed": 5,
                              "validation_metric": metric_to_optimize,
                              }

   ##########################################################################################################

    try:

        for idx, rec in enumerate(recommenders):
            if idx == 0:
                output_file_name_root = recommenders[idx].RECOMMENDER_NAME
            else:
                output_file_name_root = output_file_name_root + '_' + recommenders[idx].RECOMMENDER_NAME

        hyperparameterSearch = SearchBayesianSkopt(ScoresHybridRecommender, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

        hyperparameters_range_dictionary = {}

        hyperparameters_range_dictionary["alpha"] = Real(low = 0.1, high = 0.9, prior = 'uniform')
        hyperparameters_range_dictionary["beta"] = Real(low = 0.1, high = 0.9, prior = 'uniform')
        hyperparameters_range_dictionary["gamma"] = Real(low = 0.1, high = 0.9, prior = 'uniform')
        hyperparameters_range_dictionary["delta"] = Real(low = 0.1, high = 0.9, prior = 'uniform')


        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, recommenders, len(recommenders)],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {}
        )


        if URM_train_last_test is not None:
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
        else:
            recommender_input_args_last_test = None


        hyperparameterSearch.search(recommender_input_args,
                                    hyperparameter_search_space = hyperparameters_range_dictionary,
                                    n_cases = n_cases,
                                    n_random_starts = n_random_starts,
                                    resume_from_saved = resume_from_saved,
                                    save_model = save_model,
                                    evaluate_on_test = evaluate_on_test,
                                    max_total_time = max_total_time,
                                    output_folder_path = output_folder_path,
                                    output_file_name_root = output_file_name_root,
                                    metric_to_optimize = metric_to_optimize,
                                    cutoff_to_optimize = cutoff_to_optimize,
                                    recommender_input_args_last_test = recommender_input_args_last_test)

        



    except Exception as e:

        print("On recommender {} Exception {}".format(ScoresHybridRecommender.RECOMMENDER_NAME, str(e)))
        traceback.print_exc()

        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(ScoresHybridRecommender.RECOMMENDER_NAME, str(e)))
        error_file.close()