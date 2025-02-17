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
        "pyyaml",
        'alembic>=1.7.0',
    ],
    entry_points={
        'console_scripts': [
            'db-migrate=src.database.migrations.cli:migrate',
            'db-upgrade=src.database.migrations.cli:upgrade',
            'db-downgrade=src.database.migrations.cli:downgrade',
        ],
    },
) 