import config
import embeddings
import tensorflow as tf
import numpy as np
import json

con = config.Config()
con.set_in_path("./data/raw/FB13/")
con.set_test_link_prediction(True)
con.set_test_triple_classification(True)
con.set_work_threads(4)
con.set_dimension(50)
con.set_import_files("./models/1559927586/model.vec.tf")
con.init()
con.set_model(embeddings.TransE)
con.test()