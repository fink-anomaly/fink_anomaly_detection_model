import pickle
import os.path
import argparse
from collections import defaultdict
from functools import lru_cache
from random import randint, choice
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType


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
    return clf.best_estimator_


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
    return roc_auc_score(y_test, y_score)


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
    parser.add_argument('dataset_dir', type=str, help='Input dir for dataset')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of threads (default: -1)')
    args = parser.parse_args()
    train_data_path = args.dataset_dir
    n_jobs = args.n_jobs
    assert os.path.isfile(train_data_path), 'The specified training dataset file does not exist!'
    filter_base = ('_r', '_g')
    print('Loading training data...')
    x_buf_data = pd.read_parquet(train_data_path)
    print(f'data shape: {x_buf_data.shape}')
    features_1 = x_buf_data["lc_features"].apply(lambda data:
        extract_one(data, "1")).add_suffix("_r")
    features_2 = x_buf_data["lc_features"].apply(lambda data:
        extract_one(data, "2")).add_suffix("_g")

    print('Filtering...')
    data = pd.concat([
    x_buf_data.drop(['lc_features', 'cjd', 'cfid', 'cmag', 'cerrmag', 'is_null'], axis=1),
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
        search_params_unknown = {
            'n_estimators': (100, 150, 200, 300, 500),
            'max_features':(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            'contamination': (sum(is_unknown) / len(data[key]),),
            'bootstrap': (True,),
            'max_samples': (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            'n_jobs': (n_jobs,)
        }
        forest_simp = train_with_forest(
            data[key],
            search_params_unknown,
            scorer,
            is_unknown
        )
        with open(f'forest{key}.pickle', 'wb') as handle:
            pickle.dump(forest_simp, handle)
        forest_simp._max_features = 18
        initial_type = [('X', FloatTensorType([None, data[key].shape[1]]))]
        options = {id(forest_simp): {
            'score_samples': True
        }}
        onx = to_onnx(forest_simp, initial_types=initial_type, options=options)
        with open(f"forest{key}.onnx", "wb") as file:
            file.write(onx.SerializeToString())
