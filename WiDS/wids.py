# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:40:33 2019

@author: khira
"""

import numpy as np
from pathlib import Path
import pandas as pd

import fastai
from fastai.vision import *

work_dir = Path('')
path = Path('')

train = 'train_images'
test =  path/'leaderboard_test_data'
holdout = path/'leaderboard_holdout_data'
sample_sub = path/'SampleSubmission.csv'
labels = path/'traininglabels.csv'

df = pd.read_csv(labels)

#filteredTrain = df[df.score>0.75]

test_names = [f for f in test.iterdir()]
holdout_names = [f for f in holdout.iterdir()]

src = (ImageItemList.from_df(df, path, folder=train)
      .random_split_by_pct(0.2, seed=0)
      .label_from_df('has_oilpalm')
      .add_test(test_names+holdout_names))

data =  (src.transform(get_transforms(), size=128)
         .databunch(bs=64)
         .normalize(imagenet_stats))
#data.show_batch(rows=3, figsize=(7, 8))

#This was working perfectly some minutes ago!
from sklearn.metrics import roc_auc_score
def auc_score(preds,targets):
    return torch.tensor(roc_auc_score(targets,preds[:,1]))

learn = create_cnn(data, models.resnet18, metrics=[auc_score])

learn.lr_find()
learn.recorder.plot()
