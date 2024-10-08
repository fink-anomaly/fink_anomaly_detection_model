import pickle
import os.path
import argparse
from collections import defaultdict
from functools import lru_cache
from random import randint, choice
import pandas as pd
import psutil
import os
import time
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from coniferest.onnx import to_onnx as to_onnx_add
from coniferest.aadforest import AADForest
from sklearn.model_selection import train_test_split
import itertools
import fink_anomaly_detection_model.reactions_reader as reactions_reader


def generate_param_comb(param_dict):
    base = itertools.product(*param_dict.values())
    columns = param_dict.keys()
    for obj in base:
        yield dict(zip(columns, obj))


def train_base_AAD(data: pd.DataFrame, train_params, scorer, y_true, use_default_model=False):
    if use_default_model:
        return AADForest().fit(data.values)
    X_train, X_test, y_train, y_test = train_test_split(
        data.values, y_true, test_size=0.2, random_state=42)
    best_est = (0, None, None)
    for cur_params in generate_param_comb(train_params):
        print(cur_params)
        forest = AADForest(**cur_params)
        forest.fit(X_train, y_train)
        cur_score = scorer(forest, X_test, y_test)
        print(cur_score)
        if cur_score > best_est[0]:
            best_est = (cur_score, forest, cur_params)
    print(f'Optimal: {best_est[2]}')
    return AADForest(**best_est[2]).fit(data.values, y_true)


def extract_one(data, key) -> pd.Series:
    """
    Function for extracting data from lc_features
    :param data: dict
                lc_features dict
    :param key: str
                Name of the extracted filter
    :return: pd.DataFrame
                Dataframe with a specific filter
    """
    series = pd.Series(data[key], dtype=float)
    return series


def train_with_forest(data, train_params, scorer_, y_true) -> IsolationForest:
    """
    Training of the IsolationForest model
    :param data: pd.DataFrame
        Training dataset
    :param train_params: dict
        Model hyperparameters
    :param scorer_: function
        Model quality evaluation function
    :param y_true:
        Target
    :return: IsolationForest
        Trained model
    """
    forest = IsolationForest()
    clf = GridSearchCV(forest, train_params, scoring=scorer_, verbose=2, cv=4)
    clf.fit(data.values, y_true)
    print(f' Optimal params: {clf.best_params_}')
    return clf.best_estimator_

def scorer_AAD(estimator, X_test, y_test):
    y_score = estimator.score_samples(X_test)
    return roc_auc_score(y_test, y_score)

def scorer(estimator, x_test, y_test):
    """
    Evaluation function
    :param estimator: sklearn.model
    :param x_test: pd.DataFrame
        Dataset with predictors
    :param y_test: pd.Series
        Target values
    :return: double
        roc_auc_score
    """
    y_score = estimator.decision_function(x_test)
    cur_score = roc_auc_score(y_test, y_score)
    return cur_score


def unknown_pref_metric(y_true, y_pred):
    """
    Recall calculation
    :param y_true: pd.series
        True target values
    :param y_pred: pd.series
        Predicted values target
    :return: double
        recall score
    """
    correct_preds_r = sum(y_true & y_pred)
    trues = sum(y_true)
    return (correct_preds_r / trues) * 100


unknown_pref_scorer = make_scorer(unknown_pref_metric, greater_is_better=True)


def get_stat_param_func(data):
    """
    Function for extracting attributes from dataframe
    :param data: pd.DataFrame
    :return: function
        Returns a function that allows extraction from the feature column of the dataframe data param attribute
    """
    @lru_cache
    def get_stat_param(feature, param):
        return getattr(data[feature], param)()
    return get_stat_param


def generate_random_rows(data, count):
    """
    :param data: pd.DataFrame
    :param count: int
    :return: dict
    """
    get_param = get_stat_param_func(data)
    rows = []
    for _ in range(count):
        row = {}
        for feature in data.columns:
            feature_mean = get_param(feature, 'mean')
            feature_std = get_param(feature, 'std')
            has_negative = get_param(feature, 'min') < 0
            mults = [-1, 1] if has_negative else [1]
            value = feature_mean + feature_std * (randint(1000, 2000) / 1000) * choice(mults)
            row[feature] = value
        rows.append(row)
    return rows


def append_rows(data, rows):
    """

    :param data: pd.DataFrame
    :param rows: dict
    :return: pd.DataFrame
    """
    return data.append(rows, ignore_index=True)


def unknown_and_custom_loss(model, x_data, true_is_anomaly):
    """

    :param model: sklearn.model
    :param x_data: pd.DataFrame
    :param true_is_anomaly: pd.DataFrame
    :return:
    """
    scores = model.score_samples(x_data)
    scores_order = scores.argsort()
    len_for_check = 3000
    found = 0

    for i in scores_order[:len_for_check]:
        if true_is_anomaly.iloc[i]:
            found += 1

    return (found / len_for_check) * 100


def extract_all(data) -> pd.Series:
    """
    Function for extracting data from lc_features
    :param data: dict
                lc_features dict
    :param key: str
                Name of the extracted filter
    :return: pd.DataFrame
                Dataframe with a specific filter
    """
    series = pd.Series(data, dtype=float)
    return series


def fink_ad_model_train():
    """
    :return: None
    The function saves 6 files in the call directory:
        forest_g.onnx - Trained model for filter _g in the format onnx
        forest_r.onnx - Trained model for filter _r in the format onnx
        forest_g.pickle - Trained model for filter _g in the format pickle
        forest_r.pickle - Trained model for filter _r in the format pickle
        _g_means.csv - mean values for filter _g
        _r_means.csv - mean values for filter _r

    """
    parser = argparse.ArgumentParser(description='Fink AD model training')
    parser.add_argument('--dataset_dir', type=str, help='Input dir for dataset', default='lc_features_20210617_photometry_corrected.parquet')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of threads (default: -1)')
    parser.add_argument('-c', help='Contamination is null')
    args = parser.parse_args()
    train_data_path = args.dataset_dir
    n_jobs = args.n_jobs
    #reactions_reader.get_reactions()
    assert os.path.exists(train_data_path), 'The specified training dataset file does not exist!'
    filter_base = ('_r', '_g')
    print('Loading training data...')
    x_buf_data = pd.read_parquet(train_data_path)
    print(f'data shape: {x_buf_data.shape}')
    if "lc_features_r" not in x_buf_data.columns:
        features_1 = x_buf_data["lc_features"].apply(lambda data:
            extract_one(data, "1")).add_suffix("_r")
        features_2 = x_buf_data["lc_features"].apply(lambda data:
            extract_one(data, "2")).add_suffix("_g")
    else:
        features_1 = x_buf_data["lc_features_r"].apply(lambda data:
            extract_all(data)).add_suffix("_r")
        features_2 = x_buf_data["lc_features_g"].apply(lambda data:
            extract_all(data)).add_suffix("_g")
        
    x_buf_data = x_buf_data.rename(columns={'finkclass':'class'}, errors='ignore')
    print('Filtering...')
    data = pd.concat([
    x_buf_data[['objectId', 'candid', 'class']],
    features_1,
    features_2,
    ], axis=1).dropna(axis=0)
    

    datasets = defaultdict(lambda: defaultdict(list))

    with tqdm(total=len(data)) as pbar:
        for _, row in data.iterrows():
            for passband in filter_base:
                new_data = datasets[passband]
                new_data['object_id'].append(row.objectId)
                new_data['class'].append(row['class'])
                for col, r_data in zip(data.columns, row):
                    if not col.endswith(passband):
                        continue
                    new_data[col[:-2]].append(r_data)
            pbar.update()

    main_data = {}
    for passband in datasets:
        new_data = datasets[passband]
        new_df = pd.DataFrame(data=new_data)
        for col in new_df.columns:
            if col in ('object_id', 'class'):
                new_df[col] = new_df[col].astype(str)
                continue
            new_df[col] = new_df[col].astype('float64')
        main_data[passband] = new_df
    data = {key : main_data[key] for key in filter_base}
    assert data['_r'].shape[1] == data['_g'].shape[1], '''Mismatch of the dimensions of r/g!'''
    classes = {filter_ : data[filter_]['class'] for filter_ in filter_base}
    common_rems = [
        'percent_amplitude',
        'linear_fit_reduced_chi2',
        'inter_percentile_range_10',
        'mean_variance',
        'linear_trend',
        'standard_deviation',
        'weighted_mean',
        'mean'
    ]
    data = {key : item.drop(labels=['object_id', 'class'] + common_rems,
                axis=1) for key, item in data.items()}
    for key, item in data.items():
        item.mean().to_csv(f'{key}_means.csv')
    print('Training...')
    for key in filter_base:
        is_unknown = classes[key] == 'Unknown'
        # search_params_unknown = {
        #     'n_estimators': (100, 150, 200, 300, 500),
        #     'max_features':(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        #     'contamination': (sum(is_unknown) / len(data[key]),),
        #     'bootstrap': (True,),
        #     'max_samples': (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        #     'n_jobs': (n_jobs,)
        # }
        # forest_simp = train_with_forest(
        #     data[key],
        #     search_params_unknown,
        #     scorer,
        #     is_unknown
        # )
        # with open(f'forest{key}.pickle', 'wb') as handle:
        #     pickle.dump(forest_simp, handle)
        # forest_simp._max_features = 18
        initial_type = [('X', FloatTensorType([None, data[key].shape[1]]))]
        # options = {id(forest_simp): {
        #     'score_samples': True
        # }}
        # onx = to_onnx(forest_simp, initial_types=initial_type, options=options, target_opset={"ai.onnx.ml": 3})
        # with open(f"forest{key}.onnx", "wb") as file:
        #     file.write(onx.SerializeToString())
        search_params_aad = {
            "n_trees": (100, 150, 200, 300, 500, 700, 1024),
            "n_subsamples": (int(obj*data[key].shape[0]) for obj in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
            "tau": (1 - sum(is_unknown) / len(data[key]), ),
            "n_jobs": (n_jobs,)
        } if not args.c else {
            "n_trees": (100, 150, 200, 300, 500, 700, 1024),
            "n_subsamples": (int(obj*data[key].shape[0]) for obj in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
            "n_jobs": (n_jobs,)
        }
        forest_simp = train_base_AAD(
            data[key],
            search_params_aad,
            scorer_AAD,
            is_unknown,
            use_default_model=True
        )
        reactions_dataset = pd.read_csv(f'reactions{key}.csv')
        reactions = reactions_dataset['class'].values
        reactions_dataset.drop(['class'], inplace=True, axis=1)
        forest_simp.fit(np.array(reactions_dataset), reactions)
        onx = to_onnx_add(forest_simp, initial_types=initial_type)
        with open(f"forest{key}_AAD.onnx", "wb") as f:
            f.write(onx.SerializeToString())


if __name__=='__main__':
    start_time = time.time()
    process = psutil.Process(os.getpid())
    fink_ad_model_train()
    end_time = time.time()
    execution_time = end_time - start_time

    # Получаем использование ОЗУ
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 ** 2)  # Преобразуем в МБ

    print(f"Время выполнения: {execution_time:.2f} секунд")
    print(f"Использование ОЗУ: {memory_usage:.2f} МБ")