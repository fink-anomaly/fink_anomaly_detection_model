from setuptools import setup, find_packages


def readme():
  with open('README.md', 'r') as f:
    return f.read()


setup(name='fink_anomaly_detection_model',
      version='0.4.80',
      description='Fink SNAD Anomaly Detection Model',
      packages=find_packages(),
      author_email='timofei.psheno@gmail.com',
      install_requires=['scikit-learn>=1.3.1', 'numpy==1.26.4', 'pandas==2.0.2',
      'tqdm>=4.65.0', 'scipy>=1.10.1', 'onnx==1.16.1',
      'skl2onnx==1.17.0', 'pyarrow>=13.0.0', 'coniferest==0.0.16', 'telethon',
      'slack_sdk', 'config', 'configparser', 'fink_science==3.13.3', 'Pillow', 'pyspark==3.1.3', 'light_curve', 'psutil', 'seaborn==0.13.2', 'matplotlib==3.9.0', 'optuna', 'requests'],
      entry_points={
        'console_scripts': [
            'fink_ad_model_train = fink_anomaly_detection_model:fink_ad_model_train',
            'get_anomaly_reactions = fink_anomaly_detection_model:get_reactions',
            'data_transform = fink_anomaly_detection_model:data_transform'
        ],
    },
    python_requires='>=3.9',
    long_description=readme(),
    long_description_content_type='text/markdown'
)
