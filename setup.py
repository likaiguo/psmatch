from setuptools import setup

dependencies = [
    'seaborn',
    'statsmodels',
    'scipy',
    'patsy',
    'matplotlib',
    'pandas',
    'numpy',
    'catboost'
]

VERSION = "0.0.1"

setup(
    name='psmatch',
    packages=['psmatch'],
    version=VERSION,
    description='Matching techniques for Observational Studies',
    author='Li kaiguo',
    author_email='likaiguo',
    url='https://github.com/likaiguo/pysmatch/tree/master',
    download_url='https://github.com/likaiguo/pysmatch/archive/{}.tar.gz'.format(VERSION),
    keywords=['logistic', 'regression', 'matching', 'observational', 'study', 'causal', 'inference', 'psmatch'],
    include_package_data=True,
    install_requires=dependencies
)
