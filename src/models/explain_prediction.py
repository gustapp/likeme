import config
import embeddings 
import tensorflow as tf
import numpy as np
import json
from restore_model import restore_model
from util.tools import get_mid2name

# Restore Embedding Model
model_info, con = restore_model('./models/FB15K/1906202000/model_info.json')
con.restore_tensorflow()

# Target Relation
target_rel = '/people/person/religion'

# Load Dictionaries
""" Load Dictionaries: 
    * relation2id
    * entity2id
"""
entity2id_path = './data/benchmarks/FB15K/entity2id.txt'
relation2id_path = './data/benchmarks/FB15K/relation2id.txt'

import pandas as pd
e2i_df = get_mid2name(entity2id_path)

r2i_df = pd.read_csv(relation2id_path, sep='\t', header=None, skiprows=[0])
r2i_df.columns = ['relation', 'id']

# Minimize search area by choosing only the nearest neighbors
eh_id = 2956 #jesus
r_id = 153 #religion
res_id = [7611] 
#judaism 7611
#protestantism 1314

""" Load Embedding """
ke = con.get_parameters()

ke_ent = ke['ent_embeddings']
ke_rel = ke['rel_embeddings']

""" head entity vector """
eh_vec = ke_ent[eh_id]
r_vec = ke_rel[r_id]
et_vec = ke_ent[res_id]

exempl_tails = con.predict_tail_entity(h=eh_id, r=r_id, k=10)

dbg = [np.mean(abs(eh_vec + r_vec - ke_ent[x])) for x in exempl_tails]

""" Generate perturbed set of instances """
import numpy as np

e = eh_vec
n_instances = 5000
dimension = model_info['dimension']
noise_rate = 0.05

e_hat = []
for i in range(0, n_instances):
  noise = np.random.normal(0,noise_rate,dimension)
  e_hat.append(e + noise)

""" Minimize search area set by relation similarity """
target_rel_split = target_rel.split('/')

target_rel_comp = ['/% s/' % i for i in target_rel_split]
del target_rel_comp[0]

relations_filtered = r2i_df[(r2i_df['relation'].str.contains('|'.join(target_rel_comp)))]

""" Minimize search area by choosing only the nearest neighbors """
head_ent = eh_id
tail_ent = 7611
rel = r_id

k_nn = 5
feats_tb = []

con.lib.importTestFiles()
con.lib.importTypeFiles()
""" discover head entity features """
for rel_id in relations_filtered['id']:
  """ Consider Inverse relations (head) """
  for side in ['tail', 'head']:
    if side == 'tail':
        feat_candidates_per_relation = con.predict_tail_entity(h=head_ent, r=rel_id, k=k_nn)
    elif side == 'head':
        feat_candidates_per_relation = con.predict_head_entity(t=head_ent, r=rel_id, k=k_nn)
    feats_per_rel = []
    """ (Optimizer) If the actual candidate is false, the next ones are necessarily false too """
    is_first = True
    first_is_true = False
    for feat_cand in feat_candidates_per_relation:
        if is_first and feat_cand != head_ent:
            if side == 'tail':
                first_is_true = con.predict_triple(h=head_ent, r=rel_id, t=feat_cand)
            elif side == 'head':
                first_is_true = con.predict_triple(h=feat_cand, r=rel_id, t=head_ent)
        if first_is_true:
            feats_per_rel.append(feat_cand)  
        is_first = False
    """ If relation is applicable """
    if len(feats_per_rel) > 0:
        feats_tb.append((rel_id, feats_per_rel, side))

""" Discover feats noised set """
e_hat_feats = []
feats_ids = []
for rel_id, k_tails, side in feats_tb:
  rel = ke_rel[rel_id]
  feats_ids.append((rel_id, side))
  labels = []
  for e_fake in e_hat:
    dist_per_inst = []
    id_per_inst = []
    """ Identify nearest entity to inference """
    for tail_id in k_tails:
      tail_cand = ke_ent[tail_id]
      if side == 'tail':
        dist = np.mean(abs(e_fake + rel - tail_cand))
      elif side == 'head':
        dist = np.mean(abs(tail_cand + rel - e_fake))
      dist_per_inst.append(dist)
      id_per_inst.append(tail_id)
    """ Classify @1 """
    tail = id_per_inst[dist_per_inst.index(min(dist_per_inst))]
    labels.append(tail)
  e_hat_feats.append(labels)
  print(str(len(e_hat_feats)))

""" Build local dataset """
feats_names = [s + '|' + r2i_df['relation'].iloc[[i]].values[0] for i,s in feats_ids]
e_hat_feats_df = pd.DataFrame(data=list(map(list,zip(*e_hat_feats))), columns=feats_names)

""" *** Interpretable Model *** """
    
#target_rel = '/people/person/religion'
label = res_id[0]

def explain(df, label, target_rel):
    
    """ Replace target tail """
    def replace_target(item, label=label):
        if item == label:
            return 1
        else:
            return 0

    target_rel = 'tail' + '|' + target_rel
    df[target_rel] = df[target_rel].apply(replace_target)
    target = df.pop(target_rel)

    """ Encode labels to categorical features """
    from sklearn.preprocessing import LabelEncoder

    intrp_label = []
    for column in df:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        intrp_label += map(lambda x: '{}:{}'.format(column, e2i_df[e2i_df['id'] == x]['name'].values[0]), list(le.classes_))

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
    from sklearn.model_selection import GridSearchCV
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

    # Feature Importance on Logit
    weights = logreg.coef_
    labels = intrp_label

    exp_df = pd.DataFrame(data={'labels': labels, 'weights': weights[0]})
    exp_df.sort_values('weights', inplace=True, ascending=False)
    return exp_df.head(10)

reasons = explain(e_hat_feats_df, label, target_rel)

print('{}'.format(reasons))