from evaluate import model_Evaluate
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
import re
from embedding import Sentencizer
import numpy as np
from tag import make_txt
import spacy
from sklearn import metrics
from sklearn.svm import SVC
from os.path import join
from tag import make_tags
nlp = spacy.load("en_core_web_lg")

#######################################Prep-rocessing####################################
# df = pd.DataFrame(pd.read_csv("data.csv", encoding='ISO-8859-1',
#                               names=['Target', 'Ids', 'Date', 'Flag', 'User', 'Text']))
# df.drop_duplicates(inplace=True, subset="Text")
# df.Text = df.Text.apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
# df.Text.apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x))
# df.Text = df.Text.apply(lambda x: re.sub(r'@mention', '', x))
# df.Text = df.Text.apply(lambda x: re.sub(r'@[A-Za-z0-9_]+', '', x))
# df.Text = df.Text.apply(lambda x: re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", x))
# df.Text = df.Text.apply(lambda x: re.sub(r'\[.*?\]', ' ', x))
# df.to_csv('data_clean.csv')


def word_embedding(token):
    """Generate an embedding for a word

    Args:
        token (string): e.g. "tomorrow"

    Returns:
        list(): A list with size 300
    """
    return nlp(token).vector


def sentence_embedding(sentence):
    """Generate a sentence embedding for a given sentence by finding the mean of each 
    scalar component of the word embedding, should have same size as a word embedding.

    Args:
        sentence (string): e.g. "I think tomorrow is going to be fun"

    Returns:
        list(): a list with size 300 (same as output of word_embedding())
    """
    if not sentence:
        return None
    sentence_embed = []
    for token in nlp(sentence):
        if np.fabs(nlp.vocab[token.text].vector).sum() > 0:
            sentence_embed.append(word_embedding(token.text))
    if sentence_embed:
        return embedding_mean(sentence_embed)

    return None


def embedding_mean(sentence_embed):
    """Finding the mean of a list of word embeddings

    Args:
        sentence_embed (list(list())): a list of word embeddings, dimension: word_count x 300

    Returns:
        list(): a list with size 300 (same as output of word_embedding())
    """
    result = [0 for i in range(len(sentence_embed[0]))]

    for i in range(len(sentence_embed[0])):
        summ = 0
        for j in range(len(sentence_embed)):
            summ += sentence_embed[j][i]
        result[i] = summ/len(sentence_embed)
    return result


df_train = pd.DataFrame(pd.read_csv("1000.csv", encoding='ISO-8859-1',
                                    names=['Target', 'Ids', 'Date', 'Flag', 'User', 'Text']))

df_test = pd.DataFrame(pd.read_json("test.json", lines=True))
####################################Make training sets####################################
x_train = []
y_train = []
for ind in df_train.index:
    if ind > 0 and df_train['Text'][ind]:
        s = sentence_embedding(make_txt(df_train['Text'][ind]))
        if s:
            x_train.append(np.array(s))
            y_train.append(df_train['Target'][ind])

x_train = np.array(x_train)


####################################Make testing sets####################################
x_test = []
y_test = []
for ind in df_test.index:
    if ind > 0 and df_test['text'][ind]:
        s = sentence_embedding(make_txt(df_test['text'][ind]))
        if s and df_test['klass'][ind] != 'neutral':
            x_test.append(np.array(s))
            if df_test['klass'][ind] == 'positive':
                y_test.append('4')
            else:
                y_test.append(0)
x_test = np.array(x_test)
x_test = np.atleast_2d(x_test)


clf = SVC()
clf.fit(x_train, y_train)

y_pred1 = clf.predict(x_test)

BNBmodel = BernoulliNB()
BNBmodel.fit(x_train, y_train)

y_pred2 = BNBmodel.predict(x_test)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred3 = logreg.predict(x_test)

model_Evaluate(logreg, x_test, y_pred3)

linearSVC = LinearSVC()
linearSVC.fit(x_train, y_train)

y_pred4 = linearSVC.predict(x_test)

model_Evaluate(linearSVC, x_test, y_pred4)
