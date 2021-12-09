import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
import numpy as np
from nltk.stem import WordNetLemmatizer


# def load_dataset(path):
#     with open(path) as f:
#         data = json.load(f)
#     return data


def make_tags(txt, use_stop=False, targeted=True):
    """Applying generate a list of tags for each vocab in the given text

    Args:
        txt (string): e.g. "I think tomorrow is going to be fun"
        use_stop (bool, optional): whether stop words are used]. Defaults to False.
        targeted (bool, optional): Whether only targeted tags need to be returned]. Defaults to True.

    Returns:
        list(tuple()): A list of tuple, first item being vocab, second item being tag
    """

    stop_words = set(stopwords.words('english'))
    tokenized = sent_tokenize(txt)
    result = []
    for i in tokenized:
        wordsList = nltk.word_tokenize(i)
        wordsList = [
            w for w in wordsList if not w in stop_words] if use_stop else wordsList
        tagged = nltk.pos_tag(wordsList)
        if targeted:
            tagged = list(filter(lambda c: is_target(c[1]), tagged))
        result.append(list(set(tagged)))  # make unique
    return result


def make_txt(txt):
    """generate a string with only tagged vocabularies

    Args:
        txt (string): [e.g. "I think tomorrow is going to be fun"]

    Returns:
        (string): [e.g. "I think tomorrow fun"]
    """
    tags = make_tags(txt, targeted=True)
    word_list = []
    if tags:
        for tups in tags[0]:
            # if label(tups[1]):
            #     word_list.append(
            #         WordNetLemmatizer().lemmatize(tups[0], label(tups[1])))
            # else:
            word_list.append(tups[0])
        return ' '.join(word_list)


def is_verb(label):
    return label in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_noun(label):
    return label in ['NN', 'NNS', 'NNP', 'NNPS']


def is_adj(label):
    return label in ['JJ', 'JJR', 'JJS']


def is_adv(label):
    return label in ['RB', 'RBR', 'RBS']


def is_neg(label):
    return label == "NEG"


def is_target(label):
    return is_verb(label) or is_noun(label) or is_adj(label) or is_adv(label) or is_neg(label)


def label(label):
    """produce alabel for lemmatizer object

    Args:
        label (string): e.g. JJ
    """
    if is_noun(label):
        return 'n'
    elif is_adv(label):
        return 'a'
    elif is_adv(label):
        return 'r'
    elif is_verb(label):
        return 'v'
    return None
