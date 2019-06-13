import config
import models
import tensorflow as tf
import numpy as np
import json

def predict_model(model_path):
    """ Load Trained model from model directory path.
    """

    # Load model info
    with open(model_info_file) as f:
        model_info = json.load(f)

    con = config.Config()
    con.set_in_path('./data/raw/{}/'.format(model_info['dataset'])
    con.set_test_link_prediction(model_info['test_link_prediction'])
    con.set_test_triple_classification(model_info['test_triple_classification'])
    con.set_work_threads(model_info['work_threads'])
    con.set_dimension(model_info['dimension'])
    con.set_import_files('./models/{}/{}/model.vec.tf'.format(model_info['dataset'], model_info['timestamp']))
    con.init()
    
    #Set the knowledge embedding model
    models_path = './src/models/embeddings/'
    models_names = [f for f in listdir(models_path) if isfile(join(models_path, f))]
    sel_model_name = '{}.py'.format(model_info['model']) 
    if not any(sel_model_name in model_name for model_name in models_names):
        raise ValueError('ERROR: Informed model `{}` is unkown'.format(model_info['model']))
    _model = getattr(__import__('embeddings'), model_info['model'])
    con.set_model(_model)

    return con
