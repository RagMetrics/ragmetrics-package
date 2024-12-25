from setuptools import setup, find_packages

setup(
    name='ragmetrics',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'requests'
    ],
    author='',
    author_email='',
    description='A package for integrating RagMetrics with LLM calls',
    url='https://github.com/RagMetrics/ragmetrics-package',
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.8',
)
