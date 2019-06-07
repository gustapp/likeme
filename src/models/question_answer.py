
import config
import embeddings
import util
import tensorflow as tf
import numpy as np
import pandas as pd
import json

class QuestionAnswerModel:

  def __init__(self, model_info):
    """ Knowledge Embedding Helper """
    con = config.Config()
    con.set_in_path(model_info.dataset_path)
    con.set_work_threads(model_info.threads)
    con.set_dimension(model_info.dimension)
    con.set_import_files(model_info.model_path)
    con.init()
    con.set_model(model_info.model)
    
    self.ke = con

    """ Load Dictionaries: 
      * relation2id
      * entity2id
    """

    self.ent2id = pd.read_csv(model_info.entity2id_path, sep='\t', header=None, skiprows=[0])
    self.ent2id.columns = ['entity', 'id']

    self.rel2id = pd.read_csv(model_info.relation2id_path, sep='\t', header=None, skiprows=[0])
    self.rel2id.columns = ['relation', 'id']
    pass
  
  def get_entity_id(self, entity):
    return self.ent2id[self.ent2id['entity'] == entity]['id'].values[0]
  
  def get_relation_id(self, relation):
    return self.rel2id[self.rel2id['relation'] == relation]['id'].values[0]

  def get_entity(self, entity_id):
    return self.ent2id[self.ent2id['id'] == entity_id]['entity'].values[0]
  
  def get_relation(self, relation_id):
    return self.rel2id[self.rel2id['id'] == relation_id]['relation'].values[0]

  def answer_question(self, head, relation, tail, n_answers=1):

    eh_id = self.get_entity_id(head)
    rel_id = self.get_relation_id(relation)
    et_id = self.get_entity_id(tail)

    """ Triple Classification """
    if head and relation and tail:
      res = self.ke.predict_triple(h=eh_id, r=r_id, t=et_id, k=n_answers)
      return True, res
    """ Tail Prediction """
    elif head and relation:
      res = self.ke.predict_tail_entity(h=eh_id, r=r_id, k=n_answers)
      return True, [self.get_entity(entity_id) for entity_id in res]
    """ Head Prediction """
    elif relation and tail:
      res = self.ke.predict_head_entity(r=r_id, t=et_id, k=n_answers)
      return True, [self.get_entity(entity_id) for entity_id in res]
    """ (Error) Missing Parameters """
    else:
      return False, []
