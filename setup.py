from setuptools import setup, find_packages


def readme():
  with open('README.md', 'r') as f:
    return f.read()


setup(name='fink_anomaly_detection_model',
      version='0.3',
      description='Fink anomaly detection model',
      packages=find_packages(),
      author_email='timofei.psheno@gmail.com',
      entry_points={
        'console_scripts': [
            'fink_ad_model_train = fink_anomaly_detection_model:fink_ad_model_train',
        ]
    }
)
