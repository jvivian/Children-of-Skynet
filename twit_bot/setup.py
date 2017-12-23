from setuptools import setup, find_packages

setup(
    name='twit-bot',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'rnaseq-lib',
        'pandas',
        'keras',
        'tensorflow',
        'numpy',
    ],
    entry_points='''
        [console_scripts]
        train-rnn=train_rnn:train
    ''',
)