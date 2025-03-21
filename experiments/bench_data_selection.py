
import sys
sys.path.append('../src/swfilter')
from models import SlicedWassersteinFilter, FastEuclidianFilter, SmartSplitSlicedWassersteinFilter
import numpy as np
import scipy as sc
from sklearn.preprocessing import StandardScaler
import sklearn as sk
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
from datetime import datetime
import torch.nn as nn
import cvxpy as cp
import mlflow.pytorch
from mlflow.models.signature import infer_signature, set_signature
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from hyperopt.base import miscs_update_idxs_vals
from hyperopt.pyll.base import dfs, as_apply
from hyperopt.pyll.stochastic import implicit_stochastic_symbols
from hyperopt import hp, fmin, tpe, anneal, Trials, STATUS_OK,  pyll
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, make_scorer
from functools import partial
from wadiroscnn import scnn




class ExhaustiveSearchError(Exception):
    pass


def validate_space_exhaustive_search(space):
    supported_stochastic_symbols = ['randint', 'quniform', 'qloguniform', 'qnormal', 'qlognormal', 'categorical']
    for node in dfs(as_apply(space)):
        if node.name in implicit_stochastic_symbols:
            if node.name not in supported_stochastic_symbols:
                raise ExhaustiveSearchError('Exhaustive search is only possible with the following stochastic symbols: ' + ', '.join(supported_stochastic_symbols))


def suggest(new_ids, domain, trials, seed, nbMaxSucessiveFailures=1000):

    # Build a hash set for previous trials
    hashset = set([hash(frozenset([(key, value[0]) if len(value) > 0 else ((key, None))
                                   for key, value in trial['misc']['vals'].items()])) for trial in trials.trials])

    rng =  np.random.default_rng(seed)#np.random.RandomState(seed)
    rval = []
    for _, new_id in enumerate(new_ids):
        newSample = False
        nbSucessiveFailures = 0
        while not newSample:
            # -- sample new specs, idxs, vals
            idxs, vals = pyll.rec_eval(
                domain.s_idxs_vals,
                memo={
                    domain.s_new_ids: [new_id],
                    domain.s_rng: rng,
                })
            new_result = domain.new_result()
            new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
            miscs_update_idxs_vals([new_misc], idxs, vals)

            # Compare with previous hashes
            h = hash(frozenset([(key, value[0]) if len(value) > 0 else (
                (key, None)) for key, value in vals.items()]))
            if h not in hashset:
                newSample = True
            else:
                # Duplicated sample, ignore
                nbSucessiveFailures += 1
            
            if nbSucessiveFailures > nbMaxSucessiveFailures:
                # No more samples to produce
                return []

        rval.extend(trials.new_trial_docs([new_id],
                                          [None], [new_result], [new_misc]))
    return rval

def objective_scnn(params, data, solver_name, experiment, dataset_name, verbose, selection_algorithm, split):
    # unwrap hyperparameters:
    max_neurons = int(params['max_neurons'])
    bias = True
    #print(np.shape(data['X_train_scaled']))
    #print(np.shape(data['Y_train_scaled']))

    training_samples = np.array(np.hstack((data['X_train_scaled'], data['Y_train_scaled']))) #verify dimensions
    #print(np.shape(training_samples))
    # print info on run
    print("------start of trial: ------")
    with mlflow.start_run() as run:
        try:

            if selection_algorithm == "lof":
                n_neighbors = int(params['n_neighbors'])
                algorithm = str(params['algorithm'])
                leaf_size = int(params['leaf_size'])
                metric = str(params['metric'])

                mlflow.log_param("n_neighbors", n_neighbors)
                mlflow.log_param("algorithm", algorithm)
                mlflow.log_param("leaf_size", leaf_size)
                mlflow.log_param("metric", metric)

                filter_algorithm = LocalOutlierFactor(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, metric=metric)
                y_pred = filter_algorithm.fit_predict(training_samples)
            elif selection_algorithm == "forest":
                n_estimators = int(params['n_estimators'])
                max_samples = float(params['max_samples'])
                contamination = float(params['contamination'])
                max_features = float(params['max_features'])

                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_samples", max_samples)
                mlflow.log_param("contamination", contamination)
                mlflow.log_param("max_features", max_features)

                filter_algorithm = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, max_features=max_features, n_jobs=-1)
                y_pred = filter_algorithm.fit_predict(training_samples)
            elif selection_algorithm == "sw":
                
                eps = float(params['eps'])
                n = int(params['n'])
                n_projections = int(params['n_projections'])
                p = float(params['p'])

                mlflow.log_param("eps", eps)
                mlflow.log_param("n", n)
                mlflow.log_param("n_projections", n_projections)
                mlflow.log_param("p", p)

                filter_algorithm = SlicedWassersteinFilter(eps=eps, n=n, n_projections=n_projections, p=p, n_jobs=-1, swtype='original')
                y_pred, vote = filter_algorithm.fit_predict(training_samples)
            elif selection_algorithm == 'sw_smartsplit':
                eps = float(params['eps'])
                n = int(params['n'])
                n_projections = int(params['n_projections'])
                p = float(params['p'])
                n_clusters = int(params['n_clusters'])
                n_splits = int(params['n_splits'])

                mlflow.log_param("eps", eps)
                mlflow.log_param("n", n)
                mlflow.log_param("n_projections", n_projections)
                mlflow.log_param("p", p)
                mlflow.log_param("n_clusters", n_clusters)
                mlflow.log_param("n_splits", n_splits)

                filter_algorithm = SmartSplitSlicedWassersteinFilter(eps=eps, n=n, n_projections=n_projections, p=p, n_jobs=-1, swtype='original', n_clusters=n_clusters, n_splits=n_splits, seed = 42)
                y_pred, vote = filter_algorithm.fit_predict(training_samples)
            elif selection_algorithm == 'fast_euclidian':
                eps = float(params['eps'])
                n = int(params['n'])
                p = float(params['p'])

                mlflow.log_param("eps", eps)
                mlflow.log_param("n", n)
                mlflow.log_param("p", p)

                filter_algorithm = FastEuclidianFilter(eps=eps, n=n, p=p, n_jobs=-1)
                y_pred, vote = filter_algorithm.fit_predict(training_samples)
            elif selection_algorithm == 'none':
                y_pred = np.ones(data['Y_train_scaled'].shape[0])
            else:
                raise ValueError("Invalid selection algorithm")
                
                
            X_train_filtered = data['X_train_scaled'][y_pred == 1]
            Y_train_filtered = data['Y_train_scaled'][y_pred == 1]
            n_outliers = np.sum(y_pred == -1)

            start_time_model = datetime.now() # start timer for whole training
            # define model
            model = scnn()
            mlflow.set_tag("model_name", f"scnn_and_{selection_algorithm}")
            mlflow.log_param("max_neurons", max_neurons)
            mlflow.log_param("bias", bias)
            mlflow.log_param("dataset", dataset_name)
            #mlflow.log_param("data", data)
            mlflow.log_param("solver", solver_name)
            mlflow.log_param('split', split)


        # train
        
            model.train(X_train=X_train_filtered, Y_train=Y_train_filtered, lamb_reg = 0, bias = bias, max_neurons=max_neurons, verbose=verbose, solver=solver_name)
            model_torch = model.get_torch_model(verbose = verbose)
            end_time_model =  datetime.now() 
            # torch model
            y_pred_train_tensor = model_torch(torch.tensor(data['X_train_scaled']))
            y_pred_train_filtered_tensor = model_torch(torch.tensor(X_train_filtered))
            y_pred_val_tensor = model_torch(torch.tensor(data['X_val_scaled']))
            y_pred_test_tensor = model_torch(torch.tensor(data['X_test_scaled']))

            y_pred_train = y_pred_train_tensor.detach().numpy()
            y_pred_train_filtered = y_pred_train_filtered_tensor.detach().numpy()
            y_pred_val = y_pred_val_tensor.detach().numpy()
            y_pred_test = y_pred_test_tensor.detach().numpy()

            mean_absolute_error_train = sk.metrics.mean_absolute_error(data['Y_train'], data['scaler_y'].inverse_transform(y_pred_train))
            mean_absolute_error_train_filtered = sk.metrics.mean_absolute_error(Y_train_filtered, data['scaler_y'].inverse_transform(y_pred_train_filtered))
            mean_absolute_error_val = sk.metrics.mean_absolute_error(data['Y_val'], data['scaler_y'].inverse_transform(y_pred_val))
            mean_absolute_error_test = sk.metrics.mean_absolute_error(data['Y_test'], data['scaler_y'].inverse_transform(y_pred_test))

            root_mean_squared_error_train = sk.metrics.root_mean_squared_error(data['Y_train'], data['scaler_y'].inverse_transform(y_pred_train))
            root_mean_squared_error_train_filtered = sk.metrics.root_mean_squared_error(Y_train_filtered, data['scaler_y'].inverse_transform(y_pred_train_filtered))
            root_mean_squared_error_val = sk.metrics.root_mean_squared_error(data['Y_val'], data['scaler_y'].inverse_transform(y_pred_val))
            root_mean_squared_error_test = sk.metrics.root_mean_squared_error(data['Y_test'], data['scaler_y'].inverse_transform(y_pred_test))
            

            # Start training and testing
            mlflow.log_metric("n_outliers", n_outliers)
            mlflow.log_metric("MAE_train", mean_absolute_error_train)
            mlflow.log_metric("MAE_train_filtered", mean_absolute_error_train_filtered)
            mlflow.log_metric('MAE_val', mean_absolute_error_val)
            mlflow.log_metric("MAE_test", mean_absolute_error_test)
            mlflow.log_metric("RMSE_train", root_mean_squared_error_train)
            mlflow.log_metric("RMSE_train_filtered", root_mean_squared_error_train_filtered)
            mlflow.log_metric("RMSE_val", root_mean_squared_error_val)
            mlflow.log_metric("RMSE_test", root_mean_squared_error_test)
            
            mlflow.log_param("training time", end_time_model - start_time_model)
            print(f'Training duration: {end_time_model - start_time_model}')

        # save model
            now_string = end_time_model.strftime("%m_%d_%Y___%H_%M_%S")
        except:
            mlflow.set_tag("model_name", f"scnn_and_{selection_algorithm}")
            mlflow.log_param("max_neurons", max_neurons)
            mlflow.log_param("bias", bias)
            mlflow.log_param("dataset", dataset_name)
            #mlflow.log_param("data", data)
            mlflow.log_param("solver", solver_name)
            mlflow.log_param('split', split)
            # Start training and testing
           
            mlflow.log_metric("MAE_train", np.nan)
            mlflow.log_metric("MAE_val", 100000.0)
            mlflow.log_metric("MAE_train_filtered", np.nan)
            mlflow.log_metric("MAE_test", np.nan)
            mlflow.log_metric("RMSE_train", np.nan)
            mlflow.log_metric("RMSE_train_filtered", np.nan)
            mlflow.log_metric("RMSE_test", np.nan)
            mean_absolute_error_val = 100000.0

        mlflow.end_run() #end run
        
    return  {
         "status": STATUS_OK,
         "loss": mean_absolute_error_val
        }