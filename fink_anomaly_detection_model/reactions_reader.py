import asyncio
from telethon import TelegramClient
import re
from slack_sdk import WebClient
import io
import requests
import pandas as pd
import json
from fink_science.ad_features.processor import FEATURES_COLS
import numpy as np
import config
import argparse
import configparser
from coniferest.labels import Label


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
            ztf_id = re.findall("ZTF\S*", str(message.message))
            if len(ztf_id) == 0:
                continue
            notif_time = str(message.date)
            ztf_id = ztf_id[0]
            if not message.reactions is None:
                for obj in list(message.reactions.results):
                    if ord(obj.reaction.emoticon[0]) in reactions_good:
                        id_reacted_good.append(ztf_id)
                        print(ztf_id)
                        #print(load_on_server(ztf_id, notif_time, "ANOMALY", token))
                        break
                    elif ord(obj.reaction.emoticon[0]) in reactions_bad:
                        id_reacted_bad.append(ztf_id)
                        print(ztf_id)
                        #print(load_on_server(ztf_id, notif_time, "NOT ANOMALY", token))
                        break
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
    slack_good_reactions, slack_bad_reactions = asyncio.run(slack_signals_download(slack_token, args.slack_channel))
    print('Slack: OK')
    print('The upload is completed, generation of dataframes...')
    good_reactions = tg_good_reactions.union(slack_good_reactions)
    bad_reactions = tg_bad_reactions.union(slack_bad_reactions)
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
    for section in feature_columns:
        pdf[feature_names] = pdf[section].to_list()
        pdf_gf = pdf.drop(feature_columns, axis=1).rename(columns={'i:objectId': 'object_id'})
        pdf_gf = pdf_gf.reindex(sorted(pdf_gf.columns), axis=1)
        pdf_gf.drop(common_rems, axis=1, inplace=True)
        pdf_gf['class'] = pdf_gf['class'].apply(lambda x: Label.A if x in good_reactions else Label.R)
        pdf_gf.dropna(inplace=True)
        pdf_gf.drop_duplicates(subset=['object_id'], inplace=True)
        pdf_gf.drop(['object_id'], axis=1, inplace=True)
        pdf_gf.to_csv(f'reactions_{section[-1]}.csv', index=False)
    print('OK')

if __name__=='__main__':
    get_reactions()
