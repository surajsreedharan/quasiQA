from setuptools import setup,find_packages

setup(
    name='preprocess',
    version='0.1.0',
    packages=find_packages(),
    url='',
    license='',
    install_requires=[
        'sentence-transformers==2.0.0',
        'pandas==1.1.5',
        'apache-beam[gcp]',
        'nltk',
        'pdfminer'
    ],
    author='SurajSREEDHARAN',
    author_email='',
    description=''
)
