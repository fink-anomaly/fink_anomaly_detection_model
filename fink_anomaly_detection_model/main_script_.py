import os.path
import argparse
from collections import defaultdict
from functools import lru_cache
from random import randint, choice
import pandas as pd
import psutil
import os
from fink_science.ad_features.processor import FEATURES_COLS
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import time
import io, zipfile
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from skl2onnx.common.data_types import FloatTensorType
from coniferest.onnx import to_onnx as to_onnx_add
from coniferest.aadforest import AADForest, Label
from sklearn.model_selection import train_test_split
from functools import reduce
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any
import requests
# import fink_anomaly_detection_model.reactions_reader as reactions_reader
if __name__=='__main__':
    import reactions_reader as reactions_reader
    import gui_model_validation
else:
    import fink_anomaly_detection_model.reactions_reader as reactions_reader
    import fink_anomaly_detection_model.gui_model_validation as gui_model_validation
import json


FILTER_BASE = ('_r', '_g')


def plot_leaf_purity_ratio_histogram(
        params: Dict[str, Any],
        base_data: Dict[str, np.ndarray],
        known_features: Dict[str, np.ndarray],
        known_labels: np.ndarray,
        bins: int = 100
):
    print("Building leaf purity ratio histogram...")
    sorted_keys = sorted(known_features.keys())
    concatenated_known_features = np.hstack([known_features[key] for key in sorted_keys])
    concatenated_base_data = np.hstack(
        [base_data.get(key, np.zeros((len(concatenated_known_features), 0))) for key in sorted_keys])
    forest = AADForest(**params).fit_known(
        concatenated_base_data,
        known_data=concatenated_known_features,
        known_labels=known_labels
    )
    leaf_indices = forest.apply(concatenated_known_features)

    leaf_stats = defaultdict(lambda: {"anomalies": 0, "normals": 0})
    n_samples, n_estimators = leaf_indices.shape
    for i_sample in range(n_samples):
        label = known_labels[i_sample]
        for i_tree in range(n_estimators):
            unique_key = (i_tree, leaf_indices[i_sample, i_tree])
            if label == Label.A:
                leaf_stats[unique_key]['anomalies'] += 1
            else:
                leaf_stats[unique_key]['normals'] += 1

    total_populated_leaves = len(leaf_stats)
    print(f"  - Found {total_populated_leaves} unique populated leaves in the forest.")

    ratios = []
    infinite_ratios_count = 0

    for leaf, counts in leaf_stats.items():
        n_anomalies = counts['anomalies']
        n_normals = counts['normals']

        if n_anomalies == 0 and n_normals == 0:
            continue

        if n_normals == 0:
            if n_anomalies > 0:
                infinite_ratios_count += 1
        else:
            ratios.append(n_anomalies / n_normals)

    if not ratios and infinite_ratios_count == 0:
        print("Could not calculate any ratios. No populated leaves found.")
        return

    plt.figure(figsize=(12, 7))
    if any(r > 0 for r in ratios):
        log_bins = np.logspace(np.log10(min(r for r in ratios if r > 0) + 1e-9),
                               np.log10(max(ratios) + 1),
                               bins)
    else:  # Handle case with no positive ratios
        log_bins = bins

    plt.hist(ratios, bins=log_bins, color='royalblue', label='Leaves with Mixed Samples')
    # plt.xscale('log')

    plt.title('Distribution of (Anomalies / Normals) Ratio in Leaves', fontsize=16)
    plt.xlabel('Ratio (Anomalies / Normals) in Leaf')
    plt.ylabel('Number of Leaves')

    # Add informational text boxes
    zero_ratio_count = sum(1 for r in ratios if r == 0)

    plt.text(0.05, 0.95, f'Pure "Normal" Leaves (Ratio = 0): {zero_ratio_count}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='cornflowerblue', alpha=0.3))

    plt.text(0.05, 0.85, f'Pure "Anomaly" Leaves (Ratio = ∞): {infinite_ratios_count}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='darkorange', alpha=0.3))

    plt.text(0.05, 0.75, f'Total Populated Leaves: {total_populated_leaves}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='mediumseagreen', alpha=0.3))

    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='best')
    plt.show()


def plot_leaf_purity(
        params: Dict[str, Any],
        base_data: Dict[str, np.ndarray],
        known_features: Dict[str, np.ndarray],
        known_labels: np.ndarray,
        top_n_leaves: int = 500
) -> Dict[tuple, Dict[str, int]]:
    if not known_features:
        raise ValueError("Словарь `known_features` не может быть пустым.")

    sorted_keys = sorted(known_features.keys())
    concatenated_known_features = np.hstack([known_features[key] for key in sorted_keys])
    concatenated_base_data = np.hstack(
        [base_data.get(key, np.zeros((len(concatenated_known_features), 0))) for key in sorted_keys])

    forest = AADForest(**params).fit_known(
        concatenated_base_data,
        known_data=concatenated_known_features,
        known_labels=known_labels
    )
    leaf_indices = forest.apply(concatenated_known_features)
    leaf_stats = defaultdict(lambda: {"anomalies": 0, "normals": 0, "total": 0})
    n_samples, n_estimators = leaf_indices.shape

    for i_sample in range(n_samples):
        label = known_labels[i_sample]
        for i_tree in range(n_estimators):
            leaf_id = leaf_indices[i_sample, i_tree]
            unique_key = (i_tree, leaf_id)

            if label == Label.A:
                leaf_stats[unique_key]['anomalies'] += 1
            else:
                leaf_stats[unique_key]['normals'] += 1
            leaf_stats[unique_key]['total'] += 1

    populated_leaves = {key: val for key, val in leaf_stats.items() if val['total'] > 0}

    if not populated_leaves:
        print("Не найдено ни одного населенного листа. График не может быть построен.")
        return {}
    sorted_leaves_items = sorted(
        populated_leaves.items(),
        key=lambda item: item[1]['total'],  # Сортируем по 'total'
        reverse=True
    )

    if top_n_leaves is not None:
        sorted_leaves_items = sorted_leaves_items[:top_n_leaves]

    x_labels = [f"({key[0]}, {key[1]})" for key, val in sorted_leaves_items]  # (дерево, лист)
    anomaly_counts = [val['anomalies'] for key, val in sorted_leaves_items]
    normal_counts = [-val['normals'] for key, val in sorted_leaves_items]
    x_pos = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(max(15, len(x_labels) * 0.3), 8))

    ax.bar(x_pos, anomaly_counts, color='darkorange', label='Anomalies')
    ax.bar(x_pos, normal_counts, color='cornflowerblue', label='Normals')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Number of objects in the leaf')
    ax.set_xlabel('(tree, lv)', fontsize=8)
    ax.set_title(f'Distribution of anomalies and "non-anomalies" by the most populated leaves', fontsize=16)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=9)

    ticks = ax.get_yticks()
    ax.set_yticklabels([int(abs(tick)) for tick in ticks])

    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return dict(sorted_leaves_items)


def plot_train_test_score_distributions(
        params: Dict[str, Any],
        base_data: Dict[str, np.ndarray],
        known_features: Dict[str, np.ndarray],
        known_labels: np.ndarray,
        test_size: float = 0.5,
        random_state: int = 42,
        anomaly_percentile: float = 5.0,
        bins: int = 50
) -> Dict[str, Any]:

    # 1. Разделение данных на train и test
    indices = np.arange(len(known_labels))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=known_labels
    )
    y_train = known_labels[train_indices]
    y_test = known_labels[test_indices]

    train_known_features = {key: val[train_indices] for key, val in known_features.items()}
    test_known_features = {key: val[test_indices] for key, val in known_features.items()}

    train_scores_list, test_scores_list, base_scores_list = [], [], []

    for key in base_data.keys():
        forest = AADForest(**params).fit_known(
            base_data[key], known_data=train_known_features[key], known_labels=y_train
        )
        train_scores_list.append(forest.score_samples(train_known_features[key]))
        test_scores_list.append(forest.score_samples(test_known_features[key]))
        base_scores_list.append(forest.score_samples(base_data[key]))

    final_train_scores = np.sum(train_scores_list, axis=0)
    final_test_scores = np.sum(test_scores_list, axis=0)
    final_base_scores = np.sum(base_scores_list, axis=0)
    threshold = np.percentile(final_base_scores, anomaly_percentile)
    scores_anomaly_test = final_test_scores[y_test == Label.A]
    scores_normal_test = final_test_scores[y_test != Label.A]
    scores_anomaly_train = final_train_scores[y_train == Label.A]
    scores_normal_train = final_train_scores[y_train != Label.A]
    if len(scores_anomaly_test) > 0 and len(scores_normal_test) > 0:
        plt.figure(figsize=(12, 7))
        plt.hist(final_base_scores, bins=bins, density=True, color='mediumseagreen', alpha=0.6, label='Base data')
        plt.hist(scores_normal_test, bins=bins, density=True, color='cornflowerblue', alpha=0.7,
                 label='(Norm)')
        plt.hist(scores_anomaly_test, bins=bins, density=True, color='darkorange', alpha=0.8,
                 label='(Anomaly)')
        plt.axvline(threshold, color='crimson', linestyle='--', linewidth=2,
                    label=f'Порог по баз. данным = {threshold:.2f}')
        plt.title('Распределение скоров на ТЕСТОВОЙ выборке', fontsize=15, fontweight='bold')
        plt.xlabel('Скоры аномальности', fontsize=12)
        plt.ylabel('Плотность распределения', fontsize=12)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("В тестовой выборке отсутствует один из классов. График для теста не построен.")

    # --- График 2: Обучающая выборка ---
    if len(scores_anomaly_train) > 0 and len(scores_normal_train) > 0:
        plt.figure(figsize=(12, 7))
        plt.hist(final_base_scores, bins=bins, density=True, color='mediumseagreen', alpha=0.6, label='Базовые данные')
        plt.hist(scores_normal_train, bins=bins, density=True, color='cornflowerblue', alpha=0.7,
                 label='Не аномалии (Norm)')
        plt.hist(scores_anomaly_train, bins=bins, density=True, color='darkorange', alpha=0.8,
                 label='Аномалии (Anomaly)')
        plt.axvline(threshold, color='crimson', linestyle='--', linewidth=2,
                    label=f'Порог по баз. данным = {threshold:.2f}')
        plt.title('Распределение скоров на ОБУЧАЮЩЕЙ выборке', fontsize=15, fontweight='bold')
        plt.xlabel('Скоры аномальности', fontsize=12)
        plt.ylabel('Плотность распределения', fontsize=12)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("В обучающей выборке отсутствует один из классов. График для трейна не построен.")

    print("Готово.")

    return {
        "test_data": {"scores_anomaly": scores_anomaly_test, "scores_normal": scores_normal_test},
        "train_data": {"scores_anomaly": scores_anomaly_train, "scores_normal": scores_normal_train},
        "base_scores": final_base_scores,
        "threshold": threshold
    }


def clean_dict_from_nans_inplace(data_dict: Dict[str, np.ndarray], reactions: np.ndarray) -> Dict[str, np.ndarray]:
    if not data_dict:
        print("Входной словарь пуст. Возвращаю как есть.")
        return {}
    all_arrays = list(data_dict.values())
    try:
        num_rows = all_arrays[0].shape[0]
        if not all(arr.shape[0] == num_rows for arr in all_arrays):
            raise ValueError("Массивы в словаре имеют разное количество строк!")
    except IndexError:
        raise ValueError("Один из массивов не имеет размерности для определения строк (shape[0]).")
    rows_with_nan_mask = np.zeros(num_rows, dtype=bool)

    for arr in all_arrays:
        current_nan_mask = np.isnan(arr).any(axis=1)
        rows_with_nan_mask = np.logical_or(rows_with_nan_mask, current_nan_mask)
    keep_mask = ~rows_with_nan_mask
    original_rows = num_rows
    kept_rows = np.sum(keep_mask)
    print(f"Start count: {original_rows}")
    print(f"Count with NaN: {original_rows - kept_rows}")
    print(f"Result: {kept_rows}")
    cleaned_dict = {key: arr[keep_mask] for key, arr in data_dict.items()}

    return cleaned_dict, reactions[keep_mask]

def compare_roc_auc_aad_vs_base(
        params: Dict[str, Any],
        base_data: Dict[str, np.ndarray],
        known_features: Dict[str, np.ndarray],
        known_labels: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:

    indices = np.arange(len(known_labels))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=known_labels
    )

    y_train = known_labels[train_indices]
    y_test = known_labels[test_indices]

    train_known_features = {key: val[train_indices] for key, val in known_features.items()}
    test_known_features = {key: val[test_indices] for key, val in known_features.items()}

    aad_scores_list = []
    base_scores_list = []

    for key in base_data.keys():
        aad_forest = AADForest(**params).fit_known(
            base_data[key],
            known_data=train_known_features[key],
            known_labels=y_train
        )
        aad_scores_list.append(aad_forest.score_samples(test_known_features[key]))
        training_data_for_base_forest = base_data[key]

        base_forest = AADForest(**params).fit(training_data_for_base_forest)
        base_scores_list.append(base_forest.score_samples(test_known_features[key]))

    final_aad_scores = np.sum(aad_scores_list, axis=0)
    final_base_scores = np.sum(base_scores_list, axis=0)
    y_true_binary = (y_test == Label.A).astype(int)

    aad_y_scores_inv = -final_aad_scores
    base_y_scores_inv = -final_base_scores

    fpr_aad, tpr_aad, _ = roc_curve(y_true_binary, aad_y_scores_inv)
    auc_aad = auc(fpr_aad, tpr_aad)

    fpr_base, tpr_base, _ = roc_curve(y_true_binary, base_y_scores_inv)
    auc_base = auc(fpr_base, tpr_base)

    print(f"(AUC) for AADForest: {auc_aad:.4f}")
    print(f"(AUC) for base forest: {auc_base:.4f}")

    plt.figure(figsize=(10, 8))

    plt.plot(fpr_aad, tpr_aad, color='darkorange', lw=2,
             label=f'AADForest (AUC = {auc_aad:.3f})')

    plt.plot(fpr_base, tpr_base, color='cornflowerblue', lw=2,
             label=f'Base Forest (AUC = {auc_base:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('AADForest vs. Base Forest', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True)
    plt.show()

    results = {
        'aad': {'auc': auc_aad, 'fpr': fpr_aad, 'tpr': tpr_aad},
        'base': {'auc': auc_base, 'fpr': fpr_base, 'tpr': tpr_base}
    }
    return results


def evaluate_aadforest_params(
        params: Dict[str, Any],
        base_data: Dict[str, np.ndarray],
        known_features: Dict[str, np.ndarray],
        known_labels: np.ndarray,
        base_forest: bool = False
) -> float:
    anomaly_indices = np.where(known_labels == Label.A)[0]

    if len(anomaly_indices) == 0:
        raise ValueError()

    ranks = []
    print(f"Start validation for {len(anomaly_indices)} anomalies with: {params}")

    for i, hold_out_idx in enumerate(anomaly_indices):
        print(f"  -> Обработка аномалии {i + 1}/{len(anomaly_indices)}...")
        train_indices = np.delete(np.arange(len(known_labels)), hold_out_idx)

        def return_samples_for_key(key):
            train_known_features = known_features[key]
            train_known_labels = known_labels
            test_anomaly_features = known_features[key][hold_out_idx].reshape(1, -1)
            if base_forest:
                forest = AADForest(**params).fit(
                    base_data[key]
                )
            else:
                forest = AADForest(**params).fit_known(
                    base_data[key],
                    known_data=train_known_features[train_indices],
                    known_labels=train_known_labels[train_indices]
                )

            base_scores = forest.score_samples(base_data[key])
            hold_out_score = forest.score_samples(test_anomaly_features)[0]
            return base_scores, hold_out_score
        base_scores, hold_out_score = reduce(
            lambda base, cur: (base[0] + cur[0], base[1] + cur[1]), (return_samples_for_key(key) for key in base_data.keys())
        )
        rank = np.sum(base_scores >= hold_out_score) + 1
        ranks.append(rank)
        print(f'Rank {len(base_scores)-rank}')

    average_rank = np.median(ranks)
    print(f"Median rank: {len(base_scores)-average_rank:.2f}\n")

    return len(base_scores)-average_rank



def generate_param_comb(param_dict):
    base = itertools.product(*param_dict.values())
    columns = param_dict.keys()
    for obj in base:
        yield dict(zip(columns, obj))


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



def process_matrices(matrix1, matrix2):
    mask = ~np.isnan(matrix1).any(axis=1)
    cleaned_matrix1 = matrix1[mask]
    mean_values = np.nanmean(cleaned_matrix1, axis=0)
    nan_rows_in_matrix2 = np.isnan(matrix2).any(axis=1)
    count_nan_rows = np.sum(nan_rows_in_matrix2)
    matrix2_filled = np.where(np.isnan(matrix2), mean_values, matrix2)
    print(f"The number of lines in which NaN has been replaced: {count_nan_rows}")
    return matrix2_filled


def compare_distributions(arr1, arr2, name1='Array 1', name2='Array 2', save_dir='distribution_plots'):
    os.makedirs(save_dir, exist_ok=True)
    sns.set(style="whitegrid")

    # Histogram + KDE plot
    plt.figure(figsize=(10, 6))
    sns.histplot(arr1, kde=False, color='skyblue', label=name1, bins=50, stat='density', alpha=0.6)
    sns.histplot(arr2, kde=False, color='salmon', label=name2, bins=50, stat='density', alpha=0.6)
    plt.yscale('log')
    plt.title('Distribution Comparison')
    plt.xlabel('Score')
    plt.ylabel('Density / Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distribution_comparison.png'))
    plt.close()

    # KDE only plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(arr1, color='skyblue', label=name1, linewidth=2)
    sns.kdeplot(arr2, color='salmon', label=name2, linewidth=2)
    plt.yscale('log')
    plt.title('Kernel Density Estimation (KDE)')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kde_comparison.png'))
    plt.close()

    print(f"Plots saved in directory: {os.path.abspath(save_dir)}")

def find_or_download_file(filename, save_dir='.'):
    file_path = os.path.join(save_dir, filename)
    if os.path.isfile(file_path):
        print(f"[INFO] File '{filename}' already exists.")
        return file_path
    print(f"[INFO] File '{filename}' not found. Starting download default dataset...")
    download_url = 'https://file.cosmos.msu.ru/files/base_dataset.parquet'
    try:
        # Send a GET request with streaming
        response = requests.get(download_url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        # Get file size from headers
        total_size = int(response.headers.get('Content-Length', 0))
        downloaded_size = 0
        chunk_size = 1024
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)

                    # Display progress
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r[Download] {progress:.2f}% ({downloaded_size}/{total_size} bytes)", end='', flush=True)
                    else:
                        print(f"\r[Download] {downloaded_size} bytes", end='', flush=True)

        print(f"\n[INFO] File successfully downloaded and saved as '{filename}'.")
        return file_path

    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Error downloading file: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)  # Remove incomplete file
        return None


def plot_dirty_leaf_fraction_vs_depth(
        base_params: Dict[str, Any],
        base_data: Dict[str, np.ndarray],
        known_features: Dict[str, np.ndarray],
        known_labels: np.ndarray,
        depth_range = None
):
    """
    Plots the fraction of "dirty" leaves as a function of the `max_depth` parameter,
    using the raw, unscaled data.

    Args:
        base_params (Dict[str, Any]): Base parameters for AADForest. `max_depth` will be
                                      overwritten by the loop.
        base_data (Dict[str, np.ndarray]): The unannotated background data.
        known_features (Dict[str, np.ndarray]): The annotated feature data.
        known_labels (np.ndarray): The labels for the annotated data.
        depth_range (List[int], optional): The range of max_depth values to test.
                                           Defaults to a predefined range.
    """
    if depth_range is None:
        depth_range = list(range(12, 33, 4)) + list(range(40, 101, 10))

    print("Analyzing dirty leaf fraction vs. max_depth (without scaling)...")
    print(f"Testing depths: {depth_range}")

    # 1. Prepare data by concatenating feature sets (NO SCALING)
    print("  - Preparing data...")
    sorted_keys = sorted(known_features.keys())
    known_features_cat = np.hstack([known_features[key] for key in sorted_keys])
    base_data_cat = np.hstack([base_data.get(key, np.zeros((len(known_features_cat), 0))) for key in sorted_keys])

    dirty_fractions = []

    # 2. Loop over the specified depth range
    for i, depth in enumerate(depth_range):
        print(f"\n  -> Testing depth {depth} ({i + 1}/{len(depth_range)})...")

        current_params = base_params.copy()
        current_params['max_depth'] = depth

        # 3. Train the model on the original, unscaled data
        forest = AADForest(**current_params).fit_known(base_data_cat, known_features_cat, known_labels)

        # 4. Analyze the leaves
        leaf_indices = forest.apply(known_features_cat)
        leaf_stats = defaultdict(lambda: {"anomalies": 0, "normals": 0})

        n_samples, n_estimators = leaf_indices.shape
        for i_sample in range(n_samples):
            label = known_labels[i_sample]
            for i_tree in range(n_estimators):
                key = (i_tree, leaf_indices[i_sample, i_tree])
                if label == Label.A:
                    leaf_stats[key]['anomalies'] += 1
                else:
                    leaf_stats[key]['normals'] += 1

        # 5. Calculate the fraction
        total_populated_leaves = len(leaf_stats)
        if total_populated_leaves == 0:
            print("     -> No populated leaves found. Skipping this depth.")
            dirty_fractions.append(np.nan)
            continue

        dirty_leaves_count = sum(
            1 for stats in leaf_stats.values()
            if stats['anomalies'] > 0 and stats['normals'] > 0
        )

        fraction = dirty_leaves_count / total_populated_leaves
        dirty_fractions.append(fraction)
        print(f"     -> Total populated leaves: {total_populated_leaves}")
        print(f"     -> Dirty leaves: {dirty_leaves_count}")
        print(f"     -> Dirty Leaf Fraction: {fraction:.4f}")

    # 6. Plot the results
    plt.figure(figsize=(12, 7))
    plt.plot(depth_range, dirty_fractions, marker='o', linestyle='-', color='crimson')

    plt.title('Fraction of "Dirty" Leaves vs. Tree max_depth', fontsize=16)
    plt.xlabel('max_depth Parameter', fontsize=12)
    plt.ylabel('Fraction of Dirty Leaves', fontsize=12)
    plt.xticks(depth_range, rotation=45)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()


def fink_ad_model_train():
    """
    :return: None
    The function saves 3 files in the call directory:
        forest_g.onnx - Trained model for filter _g in the format onnx
        forest_r.onnx - Trained model for filter _r in the format onnx
        _g_means.csv - mean values for filter _g
        _r_means.csv - mean values for filter _r

    """
    parser = argparse.ArgumentParser(description='Fink AD model training')
    parser.add_argument('--dataset_dir', type=str, help='Input dir for dataset', default='lc_features_20210617_photometry_corrected.parquet')
    parser.add_argument('--load_user', type=str, default='', help='Load user from anomaly base')
    parser.add_argument('--diff', type=bool, default=False, help='Diff with base model')
    parser.add_argument('--proba_arg', type=str, default='', help='Experiment ID')
    parser.add_argument('--plot_sample', type=bool, default=False, help='Plot avg_rank(sample_factor)')
    parser.add_argument('--plot_c_a', type=bool, default=False, help='Plot avg_rank(C_a)')
    parser.add_argument('--plot_tau', type=bool, default=False, help='Plot avg_rank(tau)')
    parser.add_argument('--plot_leaf_top_pur', type=bool, default=False, help='Plot ...')
    parser.add_argument('--C_a_range', type=float, nargs=2, default=(1, 100),
                        help='C_a range for plot')
    parser.add_argument('--tau_range', type=float, nargs=2, default=(0.1, 1),
                        help='Tau range for plot')
    parser.add_argument('--sample_range', type=int, nargs=2, default=(-1, 30),
                        help='Sample factor range for plot')
    parser.add_argument('--chunk_limit', type=int, default=25,
                        help='The maximum number of objects that can be requested from Fink at a time')
    parser.add_argument('--model_test', action='store_true', help='Launch the graphical user interface for model test.')

    args = parser.parse_args()
    train_data_path = args.dataset_dir
    reactions_datasets = None
    name = None
    if args.load_user:
        reactions_datasets = reactions_reader.load_reactions(args.load_user, args.chunk_limit)
        name = args.load_user
    else:
        pass
        name = ''
        # reactions_reader.get_reactions()
    x_buf_data = pd.read_parquet(find_or_download_file(train_data_path))
    assert os.path.exists(train_data_path), 'The specified training dataset file does not exist!'
    filter_base = ('_r', '_g')
    print('Loading training data...')
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

    print('Filtering...')
    data = pd.concat([
        features_1,
        features_2,
    ], axis=1).dropna(axis=0)
    datasets = defaultdict(lambda: defaultdict(list))
    with tqdm(total=len(data)) as pbar:
        for _, row in data.iterrows():
            for passband in filter_base:
                new_data = datasets[passband]
                for col, r_data in zip(data.columns, row):
                    if not col.endswith(passband):
                        continue
                    new_data[col[:-2]].append(r_data)
            pbar.update()
    DEFAULT_PARAMS = {'n_trees': 150, 'n_subsamples': int(0.5*len(data)), 'C_a': 1000, 'budget': 100,
                      'n_jobs': None, 'random_seed': 42, 'max_depth': 28}
    main_data = {}
    for passband in datasets:
        new_data = datasets[passband]
        new_df = pd.DataFrame(data=new_data)
        for col in new_df.columns:
            new_df[col] = new_df[col].astype('float64')
        main_data[passband] = new_df
    data = {key : main_data[key] for key in filter_base}
    assert data['_r'].shape[1] == data['_g'].shape[1], '''Mismatch of the dimensions of r/g!'''
    common_rems = [
        'percent_amplitude',
        'linear_fit_reduced_chi2',
        'inter_percentile_range_10',
        'mean_variance',
        'linear_trend',
        'standard_deviation',
        'weighted_mean',
        'mean',
        # 'object_id',
        # 'class'
    ]
    data = {key : item.drop(labels=common_rems,
                axis=1) for key, item in data.items()}
    first_key = next(iter(data))
    # print(f'Используемые фичи: {data[first_key].columns}')
    for key, item in data.items():
        item.mean().to_csv(f'{key}_means.csv')
    data = {
        key : value.values.copy(order='C') for key, value in data.items()
    }
    print('Training...')
    result_models_IO = []
    target_result = None
    base_result = None
    if name:
        name = '_' + name
    filter_counter = 0
    for key in filter_base:
        initial_type = [('X', FloatTensorType([None, data[key].shape[1]]))]
        if reactions_datasets is None:
            reactions_datasets = {key : pd.read_csv(f'reactions{key}.csv') for key in filter_base}
        reactions_shapes = [dataset.shape for dataset in reactions_datasets.values()]
        if not all(reactions_dataset.shape[0] == 0 for reactions_dataset in reactions_datasets.values()):
            if filter_counter == 0:
                first_key = next(iter(reactions_datasets))
                reactions = reactions_datasets[first_key]['class'].values
                print(f'A: {np.sum(reactions==Label.A)}; R: {np.sum(reactions==Label.R)}')
                reactions_datasets = {
                    key: process_matrices(data[key], dataset.drop(['class'] + common_rems, axis=1).values).copy(order='C') for key, dataset in reactions_datasets.items()
                }
                # reactions_datasets, reactions = clean_dict_from_nans_inplace({key: dataset.drop(['class'] + common_rems, axis=1).values.copy(order='C') for key, dataset in reactions_datasets.items()}, reactions)
        else:
            reactions = np.array([])
        if args.model_test:
            gui_model_validation.launch_gui_analyzer(base_dataset=data, reactions_datasets=reactions_datasets, reactions=reactions.copy(order='C'))
            return
        print(f'Filter {key}, {len(reactions)} reactions')
        if args.plot_sample and filter_counter == 0:
            left, right = args.sample_range
            sample_factors = list(range(left, right, 1))
            result = []
            for sample_factor in sample_factors:
                anomaly_indices = np.where(reactions == Label.A)[0]
                an_features, an_reactions = {key: dataset[anomaly_indices] for key, dataset in reactions_datasets.items()}, reactions[anomaly_indices]
                nonan_features, nonan_reactions = {key: dataset[np.where(reactions != Label.A)[0]] for key, dataset in reactions_datasets.items()}, reactions[reactions != Label.A]
                nonan_count = sample_factor * len(an_reactions)
                print(nonan_count)
                if sample_factor > 0:
                    all_features, all_reactions = {key: np.vstack((an_features[key], nonan_features[key][:nonan_count])).copy(order='C') for key in data.keys()}, np.hstack((an_reactions, nonan_reactions[:nonan_count]))
                elif sample_factor == 0:
                    all_features, all_reactions = an_features, an_reactions
                else:
                    all_features, all_reactions = {key: np.vstack((an_features[key], nonan_features[key][:nonan_count])).copy(order='C') for key in data.keys()}, np.hstack((an_reactions, nonan_reactions[:nonan_count]))
                params = DEFAULT_PARAMS
                result.append(
                    evaluate_aadforest_params(
                        params=params,
                        base_data=data,
                        known_features=all_features,
                        known_labels=all_reactions.copy(order='C'),
                        base_forest=sample_factor==-1
                    )
                )
            plt.plot(sample_factors, result, marker='o')
            plt.xlabel("Sample factor")
            plt.ylabel("Median anomaly rank")
            plt.grid(True)
            plt.savefig('plot_sample.png')
            plt.close()
        if args.plot_c_a and filter_counter == 0:
            left, right = args.C_a_range
            c_a_factors = list(range(int(left), int(right), 10))
            result = []
            for c_a_factor in c_a_factors:
                params = DEFAULT_PARAMS
                params['C_a'] = c_a_factor
                result.append(
                    evaluate_aadforest_params(
                        params=params,
                        base_data=data,
                        known_features=reactions_datasets,
                        known_labels=reactions.copy(order='C')
                    )
                )
            plt.plot(c_a_factors, result, marker='o')
            plt.xlabel("C_a factor")
            plt.ylabel("Median anomaly rank")
            plt.grid(True)
            plt.savefig('plot_c_a.png')
            plt.close()
        if args.plot_tau and filter_counter == 0:
            left, right = args.tau_range
            tau_factors = np.arange(left, right, 0.1).tolist()
            result = []
            for tau_factor in tau_factors:
                params = DEFAULT_PARAMS
                params['tau'] = tau_factor
                result.append(
                    evaluate_aadforest_params(
                        params=params,
                        base_data=data,
                        known_features=reactions_datasets,
                        known_labels=reactions.copy(order='C')
                    )
                )
            plt.plot(tau_factors, result, marker='o')
            plt.xlabel("Tau factor")
            plt.ylabel("Median anomaly rank")
            plt.grid(True)
            plt.savefig('plot_tau.png')
            plt.close()
        if not filter_counter and args.plot_leaf_top_pur:
            # evaluate_aadforest_params(
            #     params=DEFAULT_PARAMS,
            #     base_data=data,
            #     known_features=reactions_datasets,
            #     known_labels=reactions.copy(order='C')
            # )
            # plot_dirty_leaf_fraction_vs_depth(
            #     base_params=DEFAULT_PARAMS,
            #     base_data=data,
            #     known_features=reactions_datasets,
            #     known_labels=reactions.copy(order='C')
            # )
            # plot_leaf_purity_ratio_histogram(
            #     params=DEFAULT_PARAMS,
            #     base_data=data,
            #     known_features=reactions_datasets,
            #     known_labels=reactions.copy(order='C')
            # )
            plot_leaf_purity(
                params=DEFAULT_PARAMS,
                base_data=data,
                known_features=reactions_datasets,
                known_labels=reactions.copy(order='C'),
            )
            # anomaly_percentile = 100 * DEFAULT_PARAMS['budget'] / 57678
            # plot_train_test_score_distributions(
            #     params=DEFAULT_PARAMS,
            #     base_data=data,
            #     known_features=reactions_datasets,
            #     known_labels=reactions.copy(order='C'),
            #     anomaly_percentile=anomaly_percentile,
            #     bins=100
            # )
            # comparison_results = compare_roc_auc_aad_vs_base(
            #     params=DEFAULT_PARAMS,
            #     base_data=data,
            #     known_features=reactions_datasets,
            #     known_labels=reactions.copy(order='C')
            # )
        forest_simp = AADForest(
            **DEFAULT_PARAMS
        ).fit_known(
            data[key],
            known_data=reactions_datasets[key],
            known_labels=reactions.copy(order='C')
        )
        if args.diff:
            base_forest = AADForest(
                **DEFAULT_PARAMS
            ).fit(data[key])
            learned_model_score = forest_simp.score_samples(data[key])
            base_model_score = base_forest.score_samples(data[key])
            compare_distributions(learned_model_score, base_model_score, name1=f'Model with {reactions_datasets[key].shape[0]} reactions', name2='Base model')
        if args.proba_arg:
            pdf = reactions_reader.get_fink_data(
                    [
                        args.proba_arg
                    ]
                )
            pdf = reactions_reader.select_best_row_per_object(pdf)
            for col in ['d:lc_features_g', 'd:lc_features_r']:
                pdf[col] = pdf[col].apply(lambda x: json.loads(x))
            feature_names = FEATURES_COLS
            pdf = pdf.loc[(pdf['d:lc_features_g'].astype(str) != '[]') & (pdf['d:lc_features_r'].astype(str) != '[]')]
            feature_columns = ['d:lc_features_g', 'd:lc_features_r']
            print(pdf.shape)
            common_rems = []
            result = dict()
            for section in feature_columns:
                pdf[feature_names] = pdf[section].to_list()
                pdf_gf = pdf.drop(feature_columns, axis=1).rename(columns={'i:objectId': 'object_id'})
                pdf_gf = pdf_gf.reindex(sorted(pdf_gf.columns), axis=1)
                pdf_gf.drop(common_rems, axis=1, inplace=True)
                pdf_gf.drop(['object_id'], axis=1, inplace=True)
                result[f'_{section[-1]}'] = pdf_gf.copy()
            base_forest = AADForest(
                **DEFAULT_PARAMS
            ).fit(data[key])

            base_model_base = base_forest.score_samples(data[key])
            base_model_target = base_forest.score_samples(result[key].values.copy(order='C'))
            base_model_score = forest_simp.score_samples(data[key])
            target_model_score = forest_simp.score_samples(result[key].values.copy(order='C'))
            print(f'Предсказан скор: {target_model_score}')
            if target_result is None:
                target_result = (base_model_score, target_model_score[0])
                base_result = (base_model_base, base_model_target[0])
            else:
                target_result = (base_model_score + target_result[0], target_result[1] + target_model_score[0])
                base_result = (base_model_base + base_result[0], base_result[1] + base_model_target[0])

        onx = to_onnx_add(forest_simp, initial_types=initial_type)
        result_models_IO.append(onx.SerializeToString())
        filter_counter += 1
    if args.proba_arg:
        sorted_data = np.sort(target_result[0])
        index = np.searchsorted(sorted_data, target_result[1]) + 1
        print(f'Model {name}, position for {args.proba_arg} is {index} from {len(sorted_data)} (model with reactions)')
        sorted_data = np.sort(base_result[0])
        index = np.searchsorted(sorted_data, base_result[1]) + 1
        print(f'Model {name}, position for {args.proba_arg} is {index} from {len(sorted_data)} (model without reactions)')
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(f'forest{FILTER_BASE[0]}_AAD{name}.onnx', result_models_IO[0])
        zip_file.writestr(f'forest{FILTER_BASE[1]}_AAD{name}.onnx', result_models_IO[1])
    zip_buffer.seek(0)
    with open(f'anomaly_detection_forest_AAD{name}.zip', 'wb') as f:
        f.write(zip_buffer.getvalue())


if __name__=='__main__':
    start_time = time.time()
    process = psutil.Process(os.getpid())
    fink_ad_model_train()
    end_time = time.time()
    execution_time = end_time - start_time

    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 ** 2)

    print(f"Время выполнения: {execution_time:.2f} секунд")
    print(f"Использование ОЗУ: {memory_usage:.2f} МБ")
