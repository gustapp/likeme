import config
import embeddings
import tensorflow as tf
import numpy as np
import json

con = config.Config()
con.set_in_path("./data/raw/FB15K/")
con.set_test_link_prediction(True)
con.set_test_triple_classification(True)
con.set_work_threads(4)
con.set_dimension(50)
con.set_import_files("./models/res/model.vec.tf")
con.init()
con.set_model(embeddings.TransE)
con.test()