import datetime
import os
import pandas as pd
import json

def generate_timestamp():
  return datetime.datetime.now().strftime('%y%m%d%H%M')

def ensure_dir(file_path):
  if not os.path.exists(file_path):
    os.makedirs(os.path.join(file_path))
    print('Creating folder: {}.'.format(file_path))
  else:
    print('Directory {} already exists!'.format(file_path))

def save_model_info(model_info, export_path):
    
  results = pd.DataFrame([model_info])

  #store embedding results within the timestamp folder
  # results.to_csv('{}/model_info.tsv'.format(export_path), sep='\t', index=False)

  with open('{}/model_info.json'.format(export_path), 'w+') as fp:
    json.dump(model_info, fp, sort_keys=True, indent=4)

  #store historic of all embedding runs
  file_to_save = os.path.expanduser('~') + '/hdd/proj/OpenKE/results/model_info_history.csv'
  if not os.path.isfile(file_to_save):
    print('creating file')
    results.to_csv(file_to_save, sep='\t', index=False)
  else:
    print('appending results to existing file')
    df = pd.read_csv(file_to_save, sep='\t')
    df = df.append(results, ignore_index=True)
    df.to_csv(file_to_save, sep='\t', index=False)

def save_training_log(training_log, export_path):
  
  results = pd.DataFrame([training_log])

  #store embedding results within the timestamp folder
  results.to_csv('{}/training_log.csv'.format(export_path), index=False)

def get_mid2name(dict_path, map_path='./mid2name.tsv'):
  """ Decode Machine Id (mid) to its correspondent name. 
      It is needed to interpret Freebase entities.
  """
    
  # Load dictionary for dataset and mid to name mapping
  entity2id_df = pd.read_csv(dict_path, sep='	', names=['mid', 'id'], skiprows=[0])
  mid2name_df = pd.read_csv('./mid2name.tsv', sep='	', names=['mid', 'name'])

  # Filter only the intersection of entity2id and mid2name to reduce computation
  mid2name_df = mid2name_df.loc[mid2name_df['mid'].isin(entity2id_df['mid'])]

  # Group multiple names for same mid (mid2name_df is now a dictionary)
  mid2name_sf = mid2name_df.groupby('mid').apply(lambda x: "%s" % '| '.join(x['name']))
  mid2name_df = pd.DataFrame({'mid':mid2name_sf.index, 'name':mid2name_sf.values})

  return pd.merge(entity2id_df, mid2name_df, how='left', on=['mid'])