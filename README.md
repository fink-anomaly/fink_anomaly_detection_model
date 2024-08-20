# Fink anomaly detection model

Здесь пока куча косяков, в обозримом будущем постараюсь их поправить

A set of modules for training models for finding anomalies in photometric data. There are currently two entry points via the console: _fink_ad_model_train_ and _get_anomaly_reactions_.

##  fink_ad_model_train

The module trains the AADForest model using expert reactions from the C055ZJJ6N2AE channels in Slack and -1001898265997 in Telegram. It creates the following files:
- _g_means.csv and _r_means.csv -- averages over the training dataset;
- _reactions_g.csv and _reactions_r.csv -- training datasets for additional training of the AADForest algorithm, based on expert reactions from Slack and Telegram channels;
- forest_g_AAD.onnx -- model for _g filter
- forest_r_AAD.onnx -- model for _r filter

**optional arguments:**

  --dataset_dir DATASET_DIR
                        Input dir for dataset (default: './lc_features_20210617_photometry_corrected.parquet')
						
  --n_jobs N_JOBS       
						Number of threads (default: -1)


**usage**: fink_ad_model_train [-h] [--dataset_dir DATASET_DIR] [--n_jobs N_JOBS]


## get_anomaly_reactions



Uploading anomaly reactions from messengers. It creates the following files:
- _reactions_g.csv and _reactions_r.csv -- training datasets for additional training of the AADForest algorithm, based on expert reactions from Slack and Telegram channels;



**optional arguments:**

  --slack_channel SLACK_CHANNEL
                        Slack Channel ID (default: 'C055ZJJ6N2AE')
  
  --tg_channel TG_CHANNEL
                        Telegram Channel ID (default: -1001898265997)

**usage**: get_anomaly_reactions [-h] [--slack_channel SLACK_CHANNEL]
                             [--tg_channel TG_CHANNEL]
