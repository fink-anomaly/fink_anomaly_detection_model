import asyncio
from telethon import TelegramClient
import re
from slack_sdk import WebClient
import io
from typing import List, Optional
import requests
import os
import pandas as pd
import json
from fink_science.ad_features.processor import FEATURES_COLS
import argparse
import configparser
from coniferest.label import Label

FILTER_BASE = ('_r', '_g')

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


def base_auth(password):
    requests.post(
    'http://157.136.253.53:24000/user/signup', json={
            'name': 'tg_data',
            'password': password
        }
    )
    r = requests.post('http://157.136.253.53:24000/user/signin', data={
    'username': 'tg_data',
    'password': password
    })
    r = json.loads(r.text)
    return f'Bearer {r["access_token"]}'


async def tg_signals_download(token, api_id, api_hash,
                                    channel_id, reactions_good={128293, 128077}, reactions_bad={128078}):
    id_reacted_good = list()
    id_reacted_bad = list()

    async with TelegramClient('reactions_session', api_id, api_hash) as client:
        async for message in client.iter_messages(channel_id):
            history_result = []
            ztf_id = re.findall("ZTF\S*", str(message.message))
            if len(ztf_id) == 0:
                continue
            notif_time = str(message.date)
            ztf_id = ztf_id[0]
            if not message.reactions is None:
                good_counter = 0
                bad_counter = 0
                for obj in list(message.reactions.results):
                    try:
                        cur_reaction = ord(obj.reaction.emoticon)
                    except TypeError:
                        print(f'not ord:{obj.reaction.emoticon}')
                    if cur_reaction in reactions_good:
                        good_counter += obj.count
                    if cur_reaction in reactions_bad:
                        bad_counter += obj.count
                print('----')
                print({obj.reaction.emoticon: obj.count for obj in list(message.reactions.results)})
                if len(list(message.reactions.results)) == 0:
                    continue
                if bad_counter >= good_counter:
                    id_reacted_bad.append(ztf_id)
                    print(f'{ztf_id}->BAD')
                    history_result.append(False)
                else:
                    id_reacted_good.append(ztf_id)
                    print(f'{ztf_id}->GOOD')
                    history_result.append(True)
            else:
                history_result.append(False)
    with open('history_list.json', 'w') as f:
        json.dump(history_result, f)
    return set(id_reacted_good), set(id_reacted_bad)



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


def get_reactions():
    config = configparser.ConfigParser()
    config.read("reactions_config.ini")
    parser = argparse.ArgumentParser(description='Uploading anomaly reactions from messengers')
    parser.add_argument('--slack_channel', type=str, help='Slack Channel ID', default='C055ZJ6N2AE')
    parser.add_argument('--tg_channel', type=int, help='Telegram Channel ID', default=-1001898265997)
    args = parser.parse_args()

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
    tg_good_reactions, tg_bad_reactions = asyncio.run(tg_signals_download('', tg_api_id, tg_api_hash, args.tg_channel))
    print('TG: OK')
    #slack_good_reactions, slack_bad_reactions = asyncio.run(slack_signals_download(slack_token, args.slack_channel))
    print('Slack: OK')
    print('The upload is completed, generation of dataframes...')
    good_reactions = tg_good_reactions.union({})
    bad_reactions = tg_bad_reactions.union({})
    oids = list(good_reactions.union(bad_reactions))
    r = requests.post(
        'https://api.fink-portal.org/api/v1/objects',
        json={
            'objectId': ','.join(oids),
            'columns': 'd:lc_features_g,d:lc_features_r,i:objectId',
            'output-format': 'json'
        }
    )
    if r.status_code != 200:
        print(r.text)
        return
    else:
        print('Fink API: OK')
    pdf = pd.read_json(io.BytesIO(r.content))
    for col in ['d:lc_features_g', 'd:lc_features_r']:
        pdf[col] = pdf[col].apply(lambda x: json.loads(x))
    feature_names = FEATURES_COLS
    pdf = pdf.loc[(pdf['d:lc_features_g'].astype(str) != '[]') & (pdf['d:lc_features_r'].astype(str) != '[]') &  ~(pdf['d:lc_features_r'].astype(str).isin('NaN') )]
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
    for section in feature_columns:
        pdf[feature_names] = pdf[section].to_list()
        pdf_gf = pdf.drop(feature_columns, axis=1).rename(columns={'i:objectId': 'object_id'})
        pdf_gf = pdf_gf.reindex(sorted(pdf_gf.columns), axis=1)
        pdf_gf.drop(common_rems, axis=1, inplace=True)
        pdf_gf['class'] = pdf_gf['object_id'].apply(lambda x: Label.A if x in good_reactions else Label.R)
        pdf_gf.dropna(inplace=True)
        pdf_gf.drop_duplicates(subset=['object_id'], inplace=True)
        pdf_gf.drop(['object_id'], axis=1, inplace=True)
        pdf_gf.to_csv(f'reactions_{section[-1]}.csv', index=False)
    print('OK')

def load_base(positive: List[str], negative: List[str]):
    print('Getting current reactions...')
    good_reactions = set(positive)
    bad_reactions = set(negative)
    oids = list(good_reactions.union(bad_reactions))
    r = requests.post(
        'https://api.fink-portal.org/api/v1/objects',
        json={
            'objectId': ','.join([obj for obj in oids if 'ZTF' in obj]),
            'columns': 'd:lc_features_g,d:lc_features_r,i:objectId',
            'output-format': 'json'
        }
    )
    if r.status_code != 200:
        print(oids)
        print(','.join([obj for obj in oids if 'ZTF' in obj]))
        print(r.text)
        return {key: pd.DataFrame() for key in FILTER_BASE}
    else:
        print('Fink API: 200')
    pdf = pd.read_json(io.BytesIO(r.content))
    if pdf.empty:
        raise Exception(f'Fink did not return any data. Most likely something is wrong: {positive}, {negative}')
    print(pdf.columns)
    real_ids = set([obj for obj in oids if 'ZTF' in obj])
    for col in ['d:lc_features_g', 'd:lc_features_r']:
        pdf[col] = pdf[col].apply(lambda x: json.loads(x))
    feature_names = FEATURES_COLS
    pdf = pdf.loc[(pdf['d:lc_features_g'].astype(str) != '[]') & (pdf['d:lc_features_r'].astype(str) != '[]') & (~pdf['d:lc_features_r'].str.contains('NaN', na=False))]
    feature_columns = ['d:lc_features_g', 'd:lc_features_r']
    print(pdf.shape)
    common_rems = []
    result = dict()
    for section in feature_columns:
        pdf[feature_names] = pdf[section].to_list()
        pdf_gf = pdf.drop(feature_columns, axis=1).rename(columns={'i:objectId': 'object_id'})
        pdf_gf = pdf_gf.reindex(sorted(pdf_gf.columns), axis=1)
        pdf_gf.drop(common_rems, axis=1, inplace=True)
        pdf_gf['class'] = pdf_gf['object_id'].apply(lambda x: Label.A if x in good_reactions else Label.R)
        pdf_gf.dropna(inplace=True)
        pdf_gf.drop_duplicates(subset=['object_id'], inplace=True)
        rec_ids = set(pdf_gf['object_id'].to_list())
        diff = real_ids.difference(rec_ids)
        if diff:
            print(f'Features not found: {diff}')
        pdf_gf.drop(['object_id'], axis=1, inplace=True)
        result[f'_{section[-1]}'] = pdf_gf.copy()
    return result

def load_reactions(name: str):
    print(f'Loading for {name}')
    service_route = f"{os.getenv('MAIN_SERVICE_URL')}/all_users_reactions"
    print(f'service_route -> {service_route}')
    resp = requests.get(service_route)
    print(f'''============
    {resp.text}
================''')
    payload = resp.json()
    for user_data in payload:
        if user_data['model_name'] == name:
            positive = user_data["positive"]
            negative = user_data["negative"]
            print(f'Получено {len(negative) + len(positive)} реакций')
            if len(negative) + len(positive) == 0:
                return {key: pd.DataFrame() for key in FILTER_BASE}
            else:
                return load_base(positive, negative)
    raise Exception('User not found in anomaly base!')


if __name__=='__main__':
    get_reactions()
