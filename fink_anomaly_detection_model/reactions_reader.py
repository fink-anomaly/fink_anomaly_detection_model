import asyncio
from telethon import TelegramClient
import re
from slack_sdk import WebClient
import io
from typing import List, Optional, Dict
import requests
import os
import pandas as pd
import json
from fink_science.ad_features.processor import FEATURES_COLS
import argparse
import configparser
from coniferest.label import Label
import time
import matplotlib.pyplot as plt
import numpy as np
from requests.exceptions import Timeout, ConnectionError
from astropy.time import Time
from datetime import datetime

FILTER_BASE = ('_r', '_g')

def get_jd_from_description(description: str) -> float | None:
    """
    Parses a description string, extracts the UTC timestamp,
    and converts it to a Julian Date.

    Args:
        description: The string containing the UTC timestamp, e.g., "... UTC: 2024-05-27 10:03:12.000 ...".

    Returns:
        The Julian Date as a float, or None if the timestamp cannot be found or parsed.
    """
    if not description:
        return None

    match = re.search(r"UTC:\s*([\d\-]+\s[\d:\.]+)", description)
    if not match:
        return None

    utc_str = match.group(1).strip()
    dt_object = None

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt_object = datetime.strptime(utc_str, fmt)
            break
        except ValueError:
            continue

    if not dt_object:
        return None

    # Конвертируем datetime в Julian Date
    return Time(dt_object, format='datetime').jd

def load_on_server(ztf_id, time, label, token):
    return requests.post(
    'http://157.136.253.53:24000/reaction/new', json={
        'ztf_id': ztf_id,
        'tag': label,
        'changed_at': time
        },
        headers={
        'Authorization': token
        }
    ).text


def status_check(res, source="not defined", timeout=60):
    if res is None or res.status_code != 200:
        url = "https://api.telegram.org/bot"
        url += os.environ["ANOMALY_TG_TOKEN"]
        method = url + "/sendMessage"
        time.sleep(8)
        requests.post(
            method,
            data={
                "chat_id": "@fink_test",
                "text": f"Source: {source}, error: {str((res.status_code if res is not None else ''))}, description: {(res.text if res is not None else '')}",
            },
            timeout=timeout,
        )
        return False
    return True


def send_post_request_with_retry(
    session: requests.Session,
    url: str,
    method: str = "POST",
    timeout=60,
    max_retries=3,
    backoff_factor=2,
    allowed_exceptions=(Timeout, ConnectionError),
    raise_on_http_error=False,
    source="not defined",
    **kwargs,
) -> requests.Response:
    for attempt in range(max_retries):
        try:
            if method == "POST":
                response = session.post(
                    url,
                    timeout=timeout,
                    **kwargs,
                )
            elif method == "GET":
                response = session.get(
                    url,
                    timeout=timeout,
                    **kwargs,
                )
            if raise_on_http_error:
                response.raise_for_status()
            else:
                status_check(response, source)
            return response

        except allowed_exceptions as e:  # noqa: PERF203
            if attempt < max_retries - 1:
                wait = backoff_factor * (2**attempt)
                status_check(
                    None,
                    f"Error: {e}. Retrying attempt {attempt + 1}/{max_retries} in {wait} seconds. ({source})",
                )
                time.sleep(wait)
            else:
                status_check(
                    None,
                    f"Failed after {max_retries} attempts. Last error: {e} ({source})",
                )
                raise
        except Exception as e:
            status_check(None, f"Unexpected error: {e} ({source})")
            raise


def get_anomalybase_userlist():
    return requests.get('https://anomaly.fink-broker.org:443/all_users_reactions').json()


def get_cutout(ztf_id):
    # transfer cutout data
    r = requests.post(
        "https://api.fink-portal.org/api/v1/cutouts",
        json={"objectId": ztf_id, "kind": "Science", "output-format": "array"},
    )
    if not status_check(r, "get cutouts"):
        return io.BytesIO()
    data = np.log(np.array(r.json()["b:cutoutScience_stampData"], dtype=float))
    plt.axis("off")
    plt.imshow(data, cmap="PuBu_r")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return buf


def get_curve(ztf_id):
    r = requests.post(
        "https://api.fink-portal.org/api/v1/objects",
        json={"objectId": ztf_id, "withupperlim": "True"},
    )
    if not status_check(r, "getting curve"):
        return None

    # Format output in a DataFrame
    pdf = pd.read_json(io.BytesIO(r.content))

    plt.figure(figsize=(15, 6))

    colordic = {1: "C0", 2: "C1"}
    filter_dict = {1: "g band", 2: "r band"}

    for filt in np.unique(pdf["i:fid"]):
        if filt == 3:
            continue
        maskFilt = pdf["i:fid"] == filt

        # The column `d:tag` is used to check data type
        maskValid = pdf["d:tag"] == "valid"
        plt.errorbar(
            pdf[maskValid & maskFilt]["i:jd"].apply(lambda x: x - 2400000.5),
            pdf[maskValid & maskFilt]["i:magpsf"],
            pdf[maskValid & maskFilt]["i:sigmapsf"],
            ls="",
            marker="o",
            color=colordic[filt],
            label=filter_dict[filt],
        )

        maskUpper = pdf["d:tag"] == "upperlim"
        plt.plot(
            pdf[maskUpper & maskFilt]["i:jd"].apply(lambda x: x - 2400000.5),
            pdf[maskUpper & maskFilt]["i:diffmaglim"],
            ls="",
            marker="^",
            color=colordic[filt],
            markerfacecolor="none",
        )

        maskBadquality = pdf["d:tag"] == "badquality"
        plt.errorbar(
            pdf[maskBadquality & maskFilt]["i:jd"].apply(lambda x: x - 2400000.5),
            pdf[maskBadquality & maskFilt]["i:magpsf"],
            pdf[maskBadquality & maskFilt]["i:sigmapsf"],
            ls="",
            marker="v",
            color=colordic[filt],
        )

    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel("Modified Julian Date")
    plt.ylabel("Difference magnitude")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf


def get_pictures(ztf_id):
    return get_cutout(ztf_id), get_curve(ztf_id)


def load_to_anomaly_base(token, ztf_id, text, username):
    base_user_list = [obj['model_name'] for obj in get_anomalybase_userlist() if obj['model_name'] == 'beta']
    timeout = 30
    for user in base_user_list:
        if user == username:
            continue
        session = requests.Session()
        res = send_post_request_with_retry(
            session=session,
            url="https://anomaly.fink-broker.org:443/user/signin",
            data={"username": user, "password": os.environ["ANOMALY_TG_TOKEN"]},
            timeout=timeout,
            source=f"load_to_anomaly_base_login_{user}",
            raise_on_http_error=True
        )
        access_token = json.loads(res.text)["access_token"]
        cutout, curve = get_pictures(ztf_id)
        cutout.seek(0)
        curve.seek(0)
        files = {"image1": cutout, "image2": curve}
        params = {"ztf_id": ztf_id}
        headers = {"Authorization": f"Bearer {access_token}"}
        text_data = json.dumps(
            {
                'user': username,
                'text': text
            }
        )
        data = {"description": text_data}
        send_post_request_with_retry(
            session=session,
            url="https://anomaly.fink-broker.org:443/images/upload",
            files=files,
            params=params,
            data=data,
            headers=headers,
            timeout=timeout,
            source="upload_to_anomaly_base",
            raise_on_http_error=True
        )


async def tg_signals_download(token, api_id, api_hash,
                              channel_id, reactions_good={128293, 128077}, reactions_bad={128078}):
    id_reacted_good = list()
    id_reacted_bad = list()
    history_result = []
    id_to_jd = {}

    async with TelegramClient('reactions_session', api_id, api_hash) as client:
        async for message in client.iter_messages(channel_id):
            ztf_id_match = re.search(r"ZTF\S*", str(message.message))
            if not ztf_id_match:
                continue
            ztf_id = ztf_id_match.group(0)
            julian_date = get_jd_from_description(message.message)
            id_to_jd[ztf_id] = julian_date
            if message.reactions:
                good_counter = 0
                bad_counter = 0
                for reaction_count in message.reactions.results:
                    try:
                        if isinstance(reaction_count.reaction.emoticon, str):
                            cur_reaction = ord(reaction_count.reaction.emoticon)
                            if cur_reaction in reactions_good:
                                good_counter += reaction_count.count
                            if cur_reaction in reactions_bad:
                                bad_counter += reaction_count.count
                    except TypeError:
                        print(f'Не удалось обработать реакцию: {reaction_count.reaction}')

                print('----')
                reactions_dict = {
                    res.reaction.emoticon: res.count
                    for res in message.reactions.results if hasattr(res.reaction, 'emoticon')
                }
                print(reactions_dict)

                if len(message.reactions.results) == 0:
                    continue

                if good_counter > 0 and token:
                    print(f'Loading to the anomaly base...')
                    load_to_anomaly_base(token, ztf_id, str(message.message), 'Citizen')

                if bad_counter >= good_counter:
                    id_reacted_bad.append(ztf_id)
                    print(f'{ztf_id}->BAD (JD: {julian_date})')
                    history_result.append(False)
                else:
                    id_reacted_good.append(ztf_id)
                    print(f'{ztf_id}->GOOD (JD: {julian_date})')
                    history_result.append(True)
            else:
                history_result.append(False)

    with open('history_list.json', 'w') as f:
        json.dump(history_result, f)

    return set(id_reacted_good), set(id_reacted_bad), id_to_jd



async def slack_signals_download(slack_token, slack_channel):
    good_react_set = {'fire', '+1'}
    bad_react_set = {'-1', 'hankey'}
    id_reacted_good = list()
    id_reacted_bad = list()
    slack_client = WebClient(slack_token)
    notif_list = slack_client.conversations_history(channel=slack_channel).__dict__['data']['messages']
    for notif in notif_list:
        if notif['type'] != 'message' or not 'text' in notif or not 'reactions' in notif:
            continue
        ztf_id = re.findall("ZTF\w*", str(notif['text']))
        if len(ztf_id) == 0:
            continue
        ztf_id = ztf_id[0]
        react_list = notif['reactions']
        for obj in react_list:
            if obj['name'] in good_react_set:
                id_reacted_good.append(ztf_id)
                break
            if obj['name'] in bad_react_set:
                id_reacted_bad.append(ztf_id)
                break
    return set(id_reacted_good), set(id_reacted_bad)


def get_fink_data(oids, chunk_limit=25):
    """
    Fetches data from Fink API for given object IDs in chunks of 100.

    Parameters:
        oids (list): List of object IDs (e.g., ZTF IDs).

    Returns:
        pd.DataFrame: Combined DataFrame with data from all successful requests.
    """

    def chunks(lst, n):
        """Yield successive n-sized chunks from a list."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # Filter only ZTF objects
    filtered_oids = [oid for oid in oids if 'ZTF' in oid]
    if not filtered_oids:
        print("No matching ZTF objects found.")
        return pd.DataFrame()

    all_data = []  # Store results from each request

    for chunk in chunks(filtered_oids, chunk_limit):
        payload = {
            'objectId': ','.join(chunk),
            'columns': 'd:lc_features_g,d:lc_features_r,i:objectId,d:anomaly_score,i:candid,i:jd',
            'output-format': 'json'
        }

        try:
            r = requests.post('https://api.fink-portal.org/api/v1/objects', json=payload)
        except Exception as e:
            print(f"Request failed: {e}")
            continue

        if r.status_code != 200:
            print(f"Error {r.status_code}: {r.text[:200]}...")  # Show short error message
            continue

        # Parse JSON response
        try:
            pdf = pd.read_json(io.BytesIO(r.content))
            all_data.append(pdf)
        except Exception as e:
            print(f"Failed to parse JSON response: {e}")

    # Combine all DataFrames
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df
    else:
        print("No data was retrieved.")
        raise Exception()


def parse_features(feature_str):
    if type(feature_str) is list:
        return feature_str
    data = feature_str.strip('[').strip(']')
    if data == '':
      return []
    data = data.split(',')
    result = []
    for obj in data:
      if obj.strip() == 'NaN':
        result.append(float('nan'))
      else:
        result.append(float(obj))
    return result


def select_best_row_per_object(df, objid_to_jd=None, ztf_to_candid_map=None):
    """
    Selects the best alert row for each object based on a hierarchy of criteria:
    1. Exact match by candid_id (if ztf_to_candid_map is provided).
    2. Closest match by Julian Date (if objid_to_jd is provided).
    3. Lowest `d:anomaly_score` (as a fallback).
    """
    if ztf_to_candid_map is None:
        ztf_to_candid_map = {}
    if objid_to_jd is None:
        objid_to_jd = {}

    results = []
    for obj_id, group in df.groupby('i:objectId'):

        known_candid = ztf_to_candid_map.get(obj_id)
        if known_candid:
            priority_row = group[group['i:candid'].astype(str) == str(known_candid)]
            if not priority_row.empty:
                results.append(priority_row.iloc[0])  # Не удаляем anomaly_score здесь
                continue
            else:
                print(f"Warning: Known candid_id '{known_candid}' for '{obj_id}' not in Fink data. Falling back.")

        cur_jd = objid_to_jd.get(obj_id)
        if cur_jd:
            time_diffs = (group['i:jd'] - cur_jd).abs()
            best_row = group.loc[time_diffs.idxmin()]
            results.append(best_row)
            continue
        if 'd:anomaly_score' in group.columns:
            valid_scores = group.dropna(subset=['d:anomaly_score'])
            if not valid_scores.empty:
                best_row = valid_scores.loc[valid_scores['d:anomaly_score'].idxmin()]
                results.append(best_row)
                continue
        print(f"Warning: No matching criteria for '{obj_id}'. Taking the first available alert.")
        results.append(group.iloc[0])

    if not results:
        return pd.DataFrame()
    final_df = pd.DataFrame(results).reset_index(drop=True)
    return final_df.drop('d:anomaly_score', axis=1, errors='ignore')


def get_reactions():
    config = configparser.ConfigParser()
    config.read("reactions_config.ini")
    parser = argparse.ArgumentParser(description='Uploading anomaly reactions from messengers')
    parser.add_argument('--slack_channel', type=str, help='Slack Channel ID', default='C055ZJ6N2AE')
    parser.add_argument('--tg_channel', type=int, help='Telegram Channel ID', default=-1001898265997)
    parser.add_argument('--cross_load', type=bool, help='Cross loading for reactions in web-service', default=False)
    args = parser.parse_args()
    cross_load = args.cross_load
    if not 'TG' in config.sections() or not 'SLACK' in config.sections():
        tg_api_id = input('Enter the TG API ID:')
        tg_api_hash = input('Enter the TG API HASH: ')
        slack_token = input('Enter the Slack token: ')
        config['TG'] = {
            'ID': tg_api_id,
            'HASH': tg_api_hash
        }
        config['SLACK'] = {'TOKEN': slack_token}
        with open('reactions_config.ini', 'w') as configfile:
            config.write(configfile)
    else:
        slack_token = config['SLACK']['TOKEN']
        tg_api_id = config['TG']['ID']
        tg_api_hash = config['TG']['HASH']
    #token = base_auth(config['BASE']['PASSWORD'])



    print('Uploading reactions from messengers...')
    if cross_load:
        token = os.environ["ANOMALY_TG_TOKEN"]
    else:
        token = ''
    tg_good_reactions, tg_bad_reactions, id_to_jd = asyncio.run(tg_signals_download(token, tg_api_id, tg_api_hash, args.tg_channel))
    print('TG: OK')
    #slack_good_reactions, slack_bad_reactions = asyncio.run(slack_signals_download(slack_token, args.slack_channel))
    print('Slack: OK')
    print('The upload is completed, generation of dataframes...')
    good_reactions = tg_good_reactions.union({})
    bad_reactions = tg_bad_reactions.union({})
    oids = list(good_reactions.union(bad_reactions))
    print(f'All {len(oids)} reactions')
    print(oids)
    debug_data = [obj for obj in oids if 'ZTF' in obj]
    pdf = get_fink_data(debug_data)
    for col in ['d:lc_features_g', 'd:lc_features_r']:
        pdf[col] = pdf[col].apply(lambda x: json.loads(x))
    feature_names = FEATURES_COLS
    pdf = pdf.loc[(pdf['d:lc_features_g'].astype(str) != '[]') & (pdf['d:lc_features_r'].astype(str) != '[]')]
    feature_columns = ['d:lc_features_g', 'd:lc_features_r']
    common_rems = [
        # 'percent_amplitude',
        # 'linear_fit_reduced_chi2',
        # 'inter_percentile_range_10',
        # 'mean_variance',
        # 'linear_trend',
        # 'standard_deviation',
        # 'weighted_mean',
        # 'mean'
    ]
    pdf = select_best_row_per_object(pdf, id_to_jd)
    for section in feature_columns:
        pdf[feature_names] = pdf[section].to_list()
        pdf_gf = pdf.drop(feature_columns, axis=1).rename(columns={'i:objectId': 'object_id'})
        pdf_gf = pdf_gf.reindex(sorted(pdf_gf.columns), axis=1)
        pdf_gf.drop(common_rems, axis=1, inplace=True)
        pdf_gf['class'] = pdf_gf['object_id'].apply(lambda x: Label.A if x in good_reactions else Label.R)
        print(f"Fink nan: {[obj for obj in debug_data if (obj not in pdf_gf['object_id'].tolist())]}")
        pdf_gf.drop(['object_id'], axis=1, inplace=True)
        pdf_gf.to_csv(f'reactions_{section[-1]}.csv', index=False)
    print('OK')


def load_base(positive: list, negative: list, chunk_limit=25) -> Dict[str, pd.DataFrame]:
    """
    Prepares a labeled dataset from Fink data.
    Works with both new (dict-based) and old (string-based) reaction formats.
    """
    all_reactions_raw = positive + negative
    if not all_reactions_raw:
        return {key: pd.DataFrame() for key in FILTER_BASE}

    print('Preparing labeled dataset...')
    print(f'Total reactions to process: {len(all_reactions_raw)}')

    is_new_format = isinstance(all_reactions_raw[0], dict)

    if is_new_format:
        print("Detected new API format (reactions as dicts with candid_id).")
        labels_df = pd.DataFrame(all_reactions_raw)
        labels_df['candid_id'] = labels_df['candid_id'].astype(str)
        positive_candids = {p['candid_id'] for p in positive if 'candid_id' in p}
        labels_df['class'] = labels_df['candid_id'].apply(lambda x: Label.A if x in positive_candids else Label.R)

        oids_to_fetch = list(labels_df['ztf_id'].unique())
        fink_df = get_fink_data(oids_to_fetch, chunk_limit)
        if fink_df.empty:
            raise Exception('Fink did not return any data for the requested ZTF objects.')
        fink_df['i:candid'] = fink_df['i:candid'].astype(str)

        pdf = pd.merge(
            fink_df, labels_df,
            left_on=['i:objectId', 'i:candid'],
            right_on=['ztf_id', 'candid_id'],
            how='inner'
        )
        print(f"Successfully matched {len(pdf)} alerts with labels.")
        if len(pdf) < len(labels_df):
            print(f"Warning: Could not find matching alerts in Fink for {len(labels_df) - len(pdf)} reactions.")

    else:
        print("Detected old API format (reactions as strings). Using fallback logic.")
        good_reactions_set = set(positive)
        bad_reactions_set = set(negative)
        oids_to_fetch = list(good_reactions_set.union(bad_reactions_set))

        fink_df = get_fink_data(oids_to_fetch, chunk_limit)
        if fink_df.empty:
            raise Exception('Fink did not return any data for the requested ZTF objects.')
        print(f'Fink return {len(fink_df)} rows')
        pdf = select_best_row_per_object(fink_df)
        pdf['class'] = pdf['i:objectId'].apply(lambda x: Label.A if x in good_reactions_set else Label.R)

    if pdf.empty:
        print("No matching data found after processing. Returning empty dataframes.")
        return {key: pd.DataFrame() for key in FILTER_BASE}

    for col in ['d:lc_features_g', 'd:lc_features_r']:
        pdf[col] = pdf[col].apply(lambda x: json.loads(x) if isinstance(x, str) else [])

    pdf = pdf.loc[
        (pdf['d:lc_features_g'].astype(str) != '[]') &
        (pdf['d:lc_features_r'].astype(str) != '[]')
        ].copy()
    print(f'{len(pdf)} left after filtering')
    feature_columns = ['d:lc_features_g', 'd:lc_features_r']
    result = dict()

    for section in feature_columns:
        pdf_copy = pdf.copy()
        pdf_copy[FEATURES_COLS] = pd.DataFrame(pdf_copy[section].to_list(), index=pdf_copy.index)

        pdf_gf = pdf_copy.drop(feature_columns, axis=1).rename(columns={'i:objectId': 'object_id'})

        pdf_gf = pdf_gf.drop(['ztf_id', 'candid_id'], axis=1, errors='ignore')

        final_columns = FEATURES_COLS + ['class']
        pdf_gf = pdf_gf[final_columns]
        result[f'_{section[-1]}'] = pdf_gf.copy()

    return result

def load_reactions(name: str, chunk_limit=25):
    print('================================')
    print(f'Loading for {name}')
    service_route = f"https://anomaly.fink-broker.org/all_users_reactions"
    print(f'service_route -> {service_route}')
    resp = requests.get(service_route)
    payload = resp.json()
    for user_data in payload:
        if user_data['model_name'] == name:
            positive = user_data["positive"]
            negative = user_data["negative"]
            print(f'{len(negative) + len(positive)} reactions')
            if len(negative) + len(positive) == 0:
                return {key: pd.DataFrame() for key in FILTER_BASE}
            else:
                return load_base(positive, negative, chunk_limit)
    raise Exception('User not found in anomaly base!')


if __name__=='__main__':
    get_reactions()
