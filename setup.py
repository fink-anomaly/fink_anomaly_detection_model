from setuptools import setup, find_packages


def readme():
  with open('README.md', 'r') as f:
    return f.read()


setup(name='fink_anomaly_detection_model',
      version='0.3.24',
      description='Fink anomaly detection model',
      packages=find_packages(),
      author_email='timofei.psheno@gmail.com',
      install_requires=['scikit-learn>=1.3.1', 'numpy>=1.23.5',
      'tqdm>=4.65.0', 'onnx==1.12.0', 'scipy>=1.10.1',
      'skl2onnx>=1.15.0', 'pyarrow>=13.0.0', 'coniferest>=0.0.11'],
      entry_points={
        'console_scripts': [
            'fink_ad_model_train = fink_anomaly_detection_model:fink_ad_model_train',
        ],
    },
    python_requires='>=3.7',
    long_description=readme(),
    long_description_content_type='text/markdown'
)
