from setuptools import setup, find_packages

setup(
    name='ragmetrics-pkg',
    version='0.1.2',
    packages=find_packages(),
    install_requires=[
        'requests'
    ],
    author='ragmetrics',
    author_email='',
    description='A package for integrating RagMetrics with LLM calls',
    url='https://github.com/RagMetrics/ragmetrics-package',
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

