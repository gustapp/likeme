# -*- coding: utf-8 -*-
import os
import click
import time
import logging
import config
import json
import embeddings
import multiprocessing
import tensorflow as tf
import numpy as np
import time as tm
from pathlib import Path
from os import listdir
from os.path import isfile, join
from dotenv import find_dotenv, load_dotenv
from ctypes import c_float
from util.tools import generate_timestamp, ensure_dir, save_model_info

@click.command()
@click.option('--model', default='TransE', help='Embedding model (e.g. TransE)')
@click.option('--dataset', '-d', default='FB15K', help='Dataset directory')
@click.option('--work_threads', '-wt', default=4, help='Work threads.')
@click.option('--train_times', '-t', default=500, help='Train times (or epochs).')
@click.option('--nbatches', '-n', default=100, help='Number of batches.')
@click.option('--alpha', '-a', default=0.001, help='Alpha.')
@click.option('--margin', '-m', default=1.0, help='Margin.')
@click.option('--bern', '-b', default=0, help='Bernoulli.')
@click.option('--dimension', '-dim', default=50, help='Dimension.')
@click.option('--ent_neg_rate', '-enr', default=1, help='Entity negative rate.')
@click.option('--rel_neg_rate', '-rnr', default=0, help='Relation negative rate.')
@click.option('--opt_method', '-opt', default='SGD', help='Optimizer (SGD or ADAGRAD).')
@click.option('--test_link_prediction', '-tlp', default=True, help='Run link prediction tests.')
@click.option('--test_triple_classification', '-ttc', default=True, help='Run triple classification tests.')
@click.option('--model_info_file', '-mi', default=None, type=click.Path(exists=True), help='Model Info File.')
def main(model, dataset, work_threads, train_times, nbatches, alpha, margin, bern, dimension, \
 ent_neg_rate, rel_neg_rate, opt_method, test_link_prediction, test_triple_classification, model_info_file):
    """ Runs model training scripts to use raw data from (../raw) 
        train the knowledge embedding model to (../models).
    """
    logger = logging.getLogger(__name__)
    logger.info('training embedding model')

    # Generate Timestamp for export path
    timestamp = generate_timestamp()
    export_path = './models/{}/{}/'.format(dataset, timestamp)

    # Initialize Model Info
    if not model_info_file:
        model_info = {
            'dataset'                   : dataset,
            'model'                     : model,
            'timestamp'                 : timestamp,
            'work_threads'              : work_threads,

            'test_link_prediction'      : test_link_prediction,
            'test_triple_classification': test_triple_classification,
            'train_times'               : train_times,
            'nbatches'                  : nbatches,
            'alpha'                     : alpha,
            'margin'                    : margin,
            'bern'                      : bern,
            'dimension'                 : dimension,
            'ent_neg_rate'              : ent_neg_rate,
            'rel_neg_rate'              : rel_neg_rate,
            'opt_method'                : opt_method
        }
    else:
        with open(model_info_file) as f:
            model_info = json.load(f)

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    # Initialize Config object
    con = config.Config()

    os.environ['CUDA_VISIBLE_DEVICES']='1'
    #Input training files from benchmarks/FB15K/ folder.
    dataset_path = './data/raw/{}/'.format(model_info['dataset'])
    if not os.path.exists(dataset_path):
        raise ValueError('ERROR: Informed dataset path `{}` could not be found.'.format(dataset_path))

    con.set_in_path(dataset_path)

    # Set Parameters
    con.set_test_link_prediction(model_info['test_link_prediction'])
    con.set_test_triple_classification(model_info['test_triple_classification'])

    con.set_work_threads(model_info['work_threads'])
    con.set_train_times(model_info['train_times'])
    con.set_nbatches(model_info['nbatches'])
    con.set_alpha(model_info['alpha'])
    con.set_margin(model_info['margin'])
    con.set_bern(model_info['bern'])
    con.set_dimension(model_info['dimension'])
    con.set_ent_neg_rate(model_info['ent_neg_rate'])
    con.set_rel_neg_rate(model_info['rel_neg_rate'])
    con.set_opt_method(model_info['opt_method'])

    #Models will be exported via tf.Saver() automatically.
    con.set_export_files(export_path + "model.vec.tf", 0)
    #Initialize experimental settings.
    con.init()
    #Set the knowledge embedding model
    models_path = './src/models/embeddings/'
    models_names = [f for f in listdir(models_path) if isfile(join(models_path, f))]
    sel_model_name = '{}.py'.format(model_info['model']) 
    if not any(sel_model_name in model_name for model_name in models_names):
        raise ValueError('ERROR: Informed model `{}` is unkown'.format(model_info['model']))
    _model = getattr(__import__('embeddings'), model_info['model'])
    con.set_model(_model)
    #Train the model.
    con.run()

    #Test model
    if model_info['test_link_prediction'] or model_info['test_triple_classification']: 
        logger.info('Testing Model')
        
        start_time = time.time()
        con.test()
        end_time = time.time()

        if model_info['test_link_prediction']: # link prediction
            model_info['mrr_raw'] = c_float.in_dll(con.lib, 'mrr_raw').value
            model_info['mr_raw'] = c_float.in_dll(con.lib, 'mr_raw').value
            model_info['hits_10_raw'] = c_float.in_dll(con.lib, 'hits_10_raw').value
            model_info['hits_3_raw'] = c_float.in_dll(con.lib, 'hits_3_raw').value
            model_info['hits_1_raw'] = c_float.in_dll(con.lib, 'hits_1_raw').value
            model_info['mrr_filter'] = c_float.in_dll(con.lib, 'mrr_filter').value
            model_info['mr_filter'] = c_float.in_dll(con.lib, 'mr_filter').value
            model_info['hits_10_filter'] = c_float.in_dll(con.lib, 'hits_10_filter').value
            model_info['hits_3_filter'] = c_float.in_dll(con.lib, 'hits_3_filter').value
            model_info['hits_1_filter'] = c_float.in_dll(con.lib, 'hits_1_filter').value

        if model_info['test_triple_classification']: # triple classification
            model_info['acc'] = c_float.in_dll(con.lib, 'aveAcc').value

        print('Model was tested in {} seconds'.format(end_time - start_time))
    save_model_info(model_info, export_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
