# from _typeshed import SupportsLenAndGetItem
from os import PathLike
import pandas as pd
import re
import numpy as np
from tag import make_txt
from tag import make_tags
import spacy


class Tweet():
    def __init__(self, Path):
        self.path = Path
        self.texts = None
        self.targets = None
        self.raw_texts = None
        self.maxLen = 0

    def check_maxLen(self, nlp, sentences):
        maxLen = 0
        for s in sentences:
            extract = make_txt(s)
            if extract:
                maxLen = max(len(list(nlp(extract))), maxLen)
        self.maxLen = maxLen

    def prepare_data(self):

        df = pd.DataFrame(pd.read_csv(self.path, encoding='ISO-8859-1',
                                      names=['Target', 'Ids', 'Date', 'Flag', 'User', 'Text']))
        nlp = spacy.load("en_core_web_lg")
        texts = []
        targets = []
        raw_texts = []
        # self.check_maxLen(nlp, list(df['Text']))

        for ind in df.index:
            if ind > 0 and df['Text'][ind]:
                # s = self.sentence_embedding(
                #     nlp, make_txt(df['Text'][ind]), self.maxLen)
                s = self.sentence_embedding(nlp, make_txt(df['Text'][ind]))
                if s:
                    texts.append(np.array(s))
                    targets.append(df['Target'][ind])
                    raw_texts.append(df['Text'][ind])
        self.texts = texts
        self.targets = targets
        self.raw_texts = raw_texts

    def word_embedding(self, nlp, token):
        """Generate an embedding for a word

        Args:
            token (string): e.g. "tomorrow"

        Returns:
            list(): A list with size 300
        """
        return nlp(token).vector

    # def sentence_embedding(self, nlp, sentence, maxLen):
    #     if not sentence:
    #         return None

    #     sentence_embed = []
    #     for token in nlp(sentence):
    #         if np.fabs(nlp.vocab[token.text].vector).sum() > 0:
    #             sentence_embed.append(self.word_embedding(nlp, token.text))
    #     sentence_embed = [j for i in sentence_embed
    #                       for j in i]
    #     for i in range(maxLen*300-len(sentence_embed)):
    #         sentence_embed.append(0)

    #     # print(type(sentence_embed))
    #     # print(len(sentence_embed))
    #     return sentence_embed

    def sentence_embedding(self, nlp, sentence):
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
                sentence_embed.append(self.word_embedding(nlp, token.text))
        if sentence_embed:
            return self.embedding_mean(sentence_embed)

        return None

    def embedding_mean(self, sentence_embed):
        """Finding the mean of a list of word embeddings

        Args:
            sentence_embed (list(list())): a list of word embeddings, dimension: word_count x 300

        Returns:
            list(): a lsist with size 300 (same as output of word_embedding())
        """
        result = [0 for i in range(len(sentence_embed[0]))]

        for i in range(len(sentence_embed[0])):
            summ = 0
            for j in range(len(sentence_embed)):
                summ += sentence_embed[j][i]
            result[i] = summ/len(sentence_embed)
        return result
