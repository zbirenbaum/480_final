from sklearn.svm import SVC
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from prepare_data import Tweet
import numpy as np
from tag import make_txt

from tag import make_tags
from evaluate import model_Evaluate
from sklearn import metrics


Tweet = Tweet('data/20000.csv')
Tweet.prepare_data()
X_train, X_test, y_train, y_test = train_test_split(
    Tweet.texts, Tweet.targets, test_size=0.2, random_state=2)  # 80% training and 20% test
clf = SVC()

clf.fit(X_train, y_train)
model_Evaluate(clf, X_test, y_test)
