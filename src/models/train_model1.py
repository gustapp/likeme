import config
import embeddings
import tensorflow as tf
import numpy as np
import time as tm

#Export Path
timestamp = int(tm.time())
export_path = './models/{}/'.format(timestamp)

con = config.Config()
#Input training files from benchmarks/FB15K/ folder.
con.set_in_path("./data/raw/FB13/")

con.set_work_threads(4)
con.set_train_times(500)
con.set_nbatches(100)
con.set_alpha(0.001)
con.set_margin(1.0)
con.set_bern(0)
con.set_dimension(50)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")

#Models will be exported via tf.Saver() automatically.
con.set_export_files(export_path + "model.vec.tf", 0)
#Model parameters will be exported to json files automatically.
con.set_out_files(export_path + "embedding.vec.json")
#Initialize experimental settings.
con.init()
#Set the knowledge embedding model
con.set_model(embeddings.TransE)
#Train the model.
con.run()