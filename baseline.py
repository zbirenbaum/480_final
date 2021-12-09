from microtc.utils import tweet_iterator
from os.path import join
from EvoMSA import base
import json
from b4msa.textmodel import TextModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from EvoMSA.utils import bootstrap_confidence_interval
from prepare_data import Tweet
from sklearn.model_selection import train_test_split
from evaluate import model_Evaluate


Tweet = Tweet('data/20000.csv')
Tweet.prepare_data()
x_train, x_test, y_train, y_test = train_test_split(
    Tweet.raw_texts, Tweet.targets, test_size=0.2, random_state=1)
evo = base.EvoMSA(TR=False, B4MSA=True, Emo=False,
                  stacked_method="sklearn.naive_bayes.GaussianNB", lang="en").fit(x_train, y_train)
model_Evaluate(evo, x_test, y_test)
