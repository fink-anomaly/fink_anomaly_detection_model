import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from functools import lru_cache, partial
from random import randint, choice
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import IsolationForest
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.utils.estimator_checks import check_estimator, check_regressors_train
from sklearn.metrics import roc_auc_score
import pickle
import sys
from skl2onnx import to_onnx
import os.path
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx

def extract_one(data, key):
    series = pd.Series(data[key], dtype=float)
    return series

def train_with_forest(data: pd.DataFrame, train_params, scorer, y_true) -> IsolationForest:
    forest = IsolationForest()
    clf = GridSearchCV(forest, train_params, scoring=scorer, verbose=2, cv=4)
    clf.fit(data.values, y_true)
    return clf.best_estimator_

def scorer(estimator, X_test, y_test):
    y_score = estimator.decision_function(X_test)
    return roc_auc_score(y_test, y_score)

def unknown_pref_metric(y_true, y_pred):
    correct_preds_r = sum(y_true & y_pred)
    trues = sum(y_true)
    return (correct_preds_r / trues) * 100


unknown_pref_scorer = make_scorer(unknown_pref_metric, greater_is_better=True)


def get_stat_param_func(data):
    @lru_cache
    def get_stat_param(feature, param):
        return getattr(data[feature], param)()
    
    return get_stat_param


def generate_random_rows(data, count):
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
    return data.append(rows, ignore_index=True)


def unknown_and_custom_loss(model, x, true_is_anomaly):
    scores = model.score_samples(x)
    scores_order = scores.argsort()
    len_for_check = 3000
    found = 0

    for i in scores_order[:len_for_check]:
        if true_is_anomaly.iloc[i]:
            found += 1

    return (found / len_for_check) * 100

def fink_ad_model_train():
    print(sys.argv)
    _, train_data_path, n_jobs = sys.argv
    n_jobs = int(n_jobs)
    assert os.path.isfile(train_data_path), 'The specified training dataset file does not exist!'
    
    print('Loading training data...')
    x = pd.read_parquet(train_data_path)
    print(f'data shape: {x.shape}')
    features_1 = x["lc_features"].apply(lambda data: extract_one(data, "1")).add_suffix("_r")
    features_2 = x["lc_features"].apply(lambda data: extract_one(data, "2")).add_suffix("_g")
    
    print('Filtering...')
    data = pd.concat([
    x.drop(['lc_features', 'cjd', 'cfid', 'cmag', 'cerrmag', 'is_null'], axis=1),
    features_1,
    features_2,
    ], axis=1).dropna(axis=0)
    
    datasets = defaultdict(lambda: defaultdict(list))
    
    with tqdm(total=len(data)) as pbar:
        for index, row in data.iterrows():
            for passband in ('_r', '_g'):
                new_data = datasets[passband]
                new_data['object_id'].append(row.objectId)
                new_data['class'].append(row['class'])
                for col, r in zip(data.columns, row):
                    if not col.endswith(passband):
                        continue
                    new_data[col[:-2]].append(r)
            pbar.update()

    main_data = dict()
    for passband in datasets:
        new_data = datasets[passband]
        new_df = pd.DataFrame(data=new_data)
        for col in new_df.columns:
            if col in ('object_id', 'class'):
                new_df[col] = new_df[col].astype(str)
                continue
            new_df[col] = new_df[col].astype('float64')
        main_data[passband] = new_df
    
    
    data_r = main_data['_r']
    data_g = main_data['_g']
    assert data_g.shape[1] == data_r.shape[1], '''Mismatch of the dimensions of r/g!'''
    
    classes_r = data_r['class']
    classes_g = data_g['class']
    data_r = data_r.drop(labels=['object_id', 'class'], axis=1)
    data_g = data_g.drop(labels=['object_id', 'class'], axis=1)
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
    data_r = data_r.drop(common_rems, axis=1)
    data_g = data_g.drop(common_rems, axis=1)
    data_r.mean().to_csv('r_means.csv')
    data_g.mean().to_csv('g_means.csv')
    is_unknown_r = classes_r == 'Unknown'
    is_unknown_g = classes_g == 'Unknown'

    search_params_unknown_r = {
        'n_estimators': (100, 150, 200, 300, 500),
        'max_features':(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        'contamination': (sum(is_unknown_r) / len(data_r),),
        'bootstrap': (True,),
        'max_samples': (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        'n_jobs': (n_jobs,)
    }

    search_params_unknown_g = {
        'n_estimators': (100, 150, 200, 300, 500),
        'max_features':(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        'contamination': (sum(is_unknown_g) / len(data_g),),
        'bootstrap': (True,),
        'max_samples': (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        'n_jobs': (n_jobs,)
    }
    print('Training...')
    forest_simp_r = train_with_forest(
        data_r,
        search_params_unknown_r,
        scorer,
        is_unknown_r
    )
    forest_simp_g = train_with_forest(
        data_g,
        search_params_unknown_g,
        scorer,
        is_unknown_g
    )
    with open('forest_g.pickle', 'wb') as handle:
        pickle.dump(forest_simp_g, handle)
    with open('forest_r.pickle', 'wb') as handle:
        pickle.dump(forest_simp_r, handle)
    forest_simp_g._max_features = 18
    forest_simp_r._max_features = 18
    initial_type_g = [('X', FloatTensorType([None, data_g.shape[1]]))]
    initial_type_r = [('X', FloatTensorType([None, data_r.shape[1]]))]
    options = {id(forest_simp_g): {
        'score_samples': True
    }}
    onx = to_onnx(forest_simp_g, initial_types=initial_type_g, options=options)
    with open("forest_g.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    options = {id(forest_simp_r): {
        'score_samples': True
    }}
    onx = to_onnx(forest_simp_r, initial_types=initial_type_r, options=options)
    with open("forest_simp_r.onnx", "wb") as f:
        f.write(onx.SerializeToString())