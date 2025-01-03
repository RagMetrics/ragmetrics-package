from setuptools import setup, find_packages

setup(
    name='ragmetrics-pkg',
    version='0.1.4',
    description='A package for integrating RagMetrics with LLM calls',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://ragmetrics.ai',    
    packages=find_packages(),
    install_requires=[
        'requests'
    ],
    author='ragmetrics',
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    project_urls={
        'Homepage': 'https://ragmetrics.ai',
        'Repository': 'https://github.com/RagMetrics/ragmetrics-package',
    },
    python_requires='>=3.6',
)

