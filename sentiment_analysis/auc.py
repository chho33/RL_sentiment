import tensorflow as tf
from tensorflow.python.platform import gfile
import random
import os
import sys
import numpy as np
import dataset
import model
import pandas as pd
from sklearn.metrics import roc_auc_score
from datetime import datetime

SEED = 112
VOCAB_SIZE = 10000
BATCH_SIZE = 32
UNIT_SIZE = 256
MAX_LENGTH = 40
CHECK_STEP = 1000.

def create_model(session, mode):
  m = model.discriminator(VOCAB_SIZE,
                          UNIT_SIZE,
                          BATCH_SIZE,
                          MAX_LENGTH,
                          mode)
  ckpt = tf.train.get_checkpoint_state('./saved_model/')

  if ckpt:
    print("Reading model from %s" % ckpt.model_checkpoint_path)
    m.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Create model with fresh parameters")
    session.run(tf.global_variables_initializer())
  return m

def evaluate():
  vocab_map, _ = dataset.read_map('corpus/mapping')
  sess = tf.Session()
  Model = create_model(sess, 'test')
  Model.batch_size = 1
  
  df = pd.read_csv('corpus/SAD.csv',header=None)
  df = df.dropna()
  #df=df.head()
  idx = list(df.index)
  random.seed(SEED)
  random.shuffle(idx)
  df = df.ix[idx]
  cut_by = int(0.9*df.shape[0])
  train_df = df.iloc[:cut_by]
  val_df = df.iloc[cut_by:] 
  for df in [train_df, val_df]:
      sentences = df[3]
      answers = df[1] 
      scores = []
      for i,sentence in enumerate(sentences):
          if i % 1000 ==0:
              print(i)
          token_ids = dataset.convert_to_token(sentence, vocab_map)
          encoder_input, encoder_length, _ = Model.get_batch([(0, token_ids)],shuffle=False) 
          score = Model.step(sess, encoder_input, encoder_length)
          #print(i,score)
          scores.append(score)
      scores = [s[0][0] for s in scores]
      auc = roc_auc_score(answers,scores)
      yield auc

train_auc,val_auc = [score for score in evaluate()]
print('train auc score: ',train_auc)
print('val auc score: ',val_auc)
with open('auc.log','a') as f:
    f.write('%s\n'%datetime.now())
    f.write('train: %s, val: %s\n'%(train_auc,val_auc))
