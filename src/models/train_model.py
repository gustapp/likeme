# -*- coding: utf-8 -*-
import click
import logging
import config
import embeddings
import tensorflow as tf
import numpy as np
import time as tm
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from ctypes import c_float


@click.command()
@click.option('--model', prompt='Your model', help='Embedding model (e.g. TransE)')
@click.option('--dataset', '-d', prompt='Your dataset path', type=click.Path(exists=True), help='Dataset directory')
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
def main(model, dataset, work_threads, train_times, nbatches, alpha, margin, bern, dimension, \
 ent_neg_rate, rel_neg_rate, opt_method, test_link_prediction, test_triple_classification):
    """ Runs model training scripts to use raw data from (../raw) 
        train the knowledge embedding model to (../models).
    """
    logger = logging.getLogger(__name__)
    logger.info('training embedding model')

    """ Generate Timestamp for export path"""
    timestamp = int(tm.time())
    export_path = './models/{}/'.format(timestamp)

    print('{}'.format(dataset))

    con = config.Config()
    #Input training files from benchmarks/FB15K/ folder.
    con.set_in_path(dataset)

    con.set_test_link_prediction(test_link_prediction)
    con.set_test_triple_classification(test_triple_classification)

    con.set_work_threads(work_threads)
    con.set_train_times(train_times)
    con.set_nbatches(nbatches)
    con.set_alpha(alpha)
    con.set_margin(margin)
    con.set_bern(bern)
    con.set_dimension(dimension)
    con.set_ent_neg_rate(ent_neg_rate)
    con.set_rel_neg_rate(rel_neg_rate)
    con.set_opt_method(opt_method)

    #Models will be exported via tf.Saver() automatically.
    con.set_export_files(export_path + "model.vec.tf", 0)
    #Model parameters will be exported to json files automatically.
    #con.set_out_files(export_path + "embedding.vec.json")
    #Initialize experimental settings.
    con.init()
    #Set the knowledge embedding model
    con.set_model(embeddings.TransE)
    #Train the model.
    con.run()

    #Test model
    con.test()

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

    if model_info['test_triple_class']: # triple classification
        model_info['acc'] = c_float.in_dll(con.lib, 'aveAcc').value
        print "\nAccuracy in test set is {}\n".format(model_info['acc'])

    model_info['testing_time'] = con.log['testing_time']
    print('Model was tested in {} seconds'.format(model_info['testing_time']))
    save_model_info()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
