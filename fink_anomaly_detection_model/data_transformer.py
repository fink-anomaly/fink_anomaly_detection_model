import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
import argparse



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



def data_transform():
    parser = argparse.ArgumentParser(description='Fink AD model transform')
    parser.add_argument('--dataset_dir', type=str, help='Input dir for dataset', default='lc_features_20210617_photometry_corrected.parquet')
    args = parser.parse_args()
    train_data_path = args.dataset_dir
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
        item.to_parquet(f'{key}_{train_data_path}.parquet')