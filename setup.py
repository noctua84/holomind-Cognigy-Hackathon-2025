from setuptools import setup, find_packages

setup(
    name="holomind",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "mlflow",
        "tensorboard",
        "pymongo",
        "psycopg2-binary",
        "pyyaml"
    ]
) 