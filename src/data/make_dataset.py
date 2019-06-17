# -*- coding: utf-8 -*-
import os
import click
import logging
import kaggle
from pathlib import Path


@click.command()
@click.option('--dataset_name', default='knowledge-graph-benchmarks', help='Kaggle dataset name to download.')
@click.option('--dataset_path', default='./data/', type=click.Path(), help='Dataset output path.')
def main(dataset_name, dataset_path):
    """ Runs data processing scripts to download raw data from Kaggle 
        to data directory (./data/...).
    """
    logger = logging.getLogger(__name__)
    logger.info('Start authenticating Kaggle account.')

    # Authenticate Kaggle account
    kaggle.api.authenticate()

    if not os.path.exists(dataset_path):
        logger.warn('Data directory not found. Creating a new one.')
        os.makedirs(dataset_path)

    kaggle.api.dataset_download_files(dataset_name, path=dataset_path, unzip=True)
    logger.info('Successfully downloaded {} dataset to {} directory.'.format(dataset_name, dataset_path))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
