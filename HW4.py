from microtc.utils import tweet_iterator
from os.path import join
import json
from b4msa.textmodel import TextModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from EvoMSA.utils import bootstrap_confidence_interval

# Reading the dataset
PATH = "."
fname = join(PATH, "semeval2017_En_train.json")
# Training set
train_data = list(tweet_iterator(fname))
test_fname = join(PATH, "test.json")
# Test set
test_data = list(tweet_iterator(test_fname))

# Code to train
tm = TextModel(lang="english", token_list=[-1], stemming=True).fit(train_data)
# le = LabelEncoder().fit([x['klass'] for x in train_data])
le = LabelEncoder().fit(['negative', 'neutral', 'positive'])


X = tm.transform(train_data)
y = le.transform([x['klass'] for x in train_data])
m = LinearSVC().fit(X, y)
# Code to predict

hy = []
for test in test_data:
    hy.append(m.predict(tm.transform(test['text'])))

# Assuming that the predictions are in an iterable variable hy
# the predictions.json file is created as follows

output = join(PATH, "predictions.json")
with open(output, 'w') as fpt:
    for x, y in zip(test_data, hy):
        x['klass'] = str(le.inverse_transform(y)[0])
        print(json.dumps(x), file=fpt)
