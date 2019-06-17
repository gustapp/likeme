import config
import embeddings
import tensorflow as tf
import numpy as np
import json
from restore_model import restore_model

# Load Embedding Model
con = restore_model('./models/FB13/1906141015/model_info.json')
# con = config.Config()
# con.set_in_path("./data/benchmarks/FB13/")
# con.set_test_link_prediction(True)
# con.set_test_triple_classification(True)
# con.set_work_threads(4)
# con.set_dimension(50)
# con.set_import_files("./models/FB13/1906141015/model.vec.tf")
# con.init()
# con.set_model(embeddings.TransE)

# Load Dictionaries
""" Load Dictionaries: 
    * relation2id
    * entity2id
"""
entity2id_path = './data/benchmarks/FB13/entity2id.txt'
relation2id_path = './data/benchmarks/FB13/relation2id.txt'

import pandas as pd
e2i_df = pd.read_csv(entity2id_path, sep='\t', header=None, skiprows=[0])
e2i_df.columns = ['entity', 'id']

r2i_df = pd.read_csv(relation2id_path, sep='\t', header=None, skiprows=[0])
r2i_df.columns = ['relation', 'id']

# Retrieve Embedding Matrix
ke = con.get_parameters()

ke_ent = ke['ent_embeddings']
ke_rel = ke['rel_embeddings']

# Minimize search area by choosing only the nearest neighbors
eh_id = 13445 #jesus
r_id = 0 #religion
res_id = [144] #judaism

""" Load Embedding """
ke = con.get_parameters()

ke_ent = ke['ent_embeddings']
ke_rel = ke['rel_embeddings']

""" head entity vector """
eh_vec = ke_ent[eh_id]

""" Generate perturbed set of instances """
import numpy as np

e = eh_vec
n_instances = 1000
dimension = 50
noise_rate = 0.03

e_hat = []
for i in range(0, n_instances):
  noise = np.random.normal(0,noise_rate,dimension)
  e_hat.append(e + noise)

""" Minimize search area by choosing only the nearest neighbors """
head_ent = eh_id
rel = r_id

k_nn = 5
feats_tb = []
""" discover head entity features """
for rel_id in [0, 3, 4, 6, 7, 12]:
  feat_candidates_per_relation = con.predict_tail_entity(h=head_ent, r=rel_id, k=k_nn)
  feats_tb.append((ke_rel[rel_id], [(ent_id, ke_ent[ent_id]) for ent_id in feat_candidates_per_relation]))

""" Discover feats noised set """
e_hat_feats = []
for rel, k_tails in feats_tb:
  labels = []
  for e_fake in e_hat:
    dist_per_inst = []
    id_per_inst = []
    """ Identify nearest entity to inference """
    for tail_id, tail_cand in k_tails:
      dist = np.mean(abs(e_fake + rel - tail_cand))
      dist_per_inst.append(dist)
      id_per_inst.append(tail_id)
    """ Classify @1 """
    tail = id_per_inst[dist_per_inst.index(min(dist_per_inst))]
    labels.append(tail)
  e_hat_feats.append(labels)
  print(str(len(e_hat_feats)))

""" Build local dataset """
feats_names = ['religion', 'profession', 'location', 'nationality', 'place_of_birth', 'ethnicity']
e_hat_feats_df = pd.DataFrame(data=list(map(list,zip(*e_hat_feats))), columns=feats_names)
    
""" *** Interpretable Model *** """
    
target_rel = 'religion'
label = res_id[0]

""" Replace target tail """
def replace_target(item, label=label):
  if item == label:
    return 1
  else:
    return 0

df = e_hat_feats_df

df[target_rel] = df[target_rel].apply(replace_target)
target = df.pop(target_rel)

""" Encode labels to categorical features """
from sklearn.preprocessing import LabelEncoder

intrp_label = []
for column in df:
  le = LabelEncoder()
  df[column] = le.fit_transform(df[column])
  intrp_label += map(lambda x: '{}:{}'.format(column, x), list(le.classes_))

""" Encode one hot """ 
from sklearn.preprocessing import OneHotEncoder

ohc = OneHotEncoder()
out = ohc.fit_transform(df)

""" Full set """
X = out
y = target

""" Logistic Regression """
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

""" Train Model """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

""" (Log) Accuracy """
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

""" (Log) Confusion Matrix """
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# Feature Importance (why jesus religion is judaism?)
weights = logreg.coef_
labels = intrp_label

exp_df = pd.DataFrame(data={'labels': labels, 'weights': weights[0]})
exp_df.sort_values('weights', inplace=True, ascending=False)
exp_df.head(3)