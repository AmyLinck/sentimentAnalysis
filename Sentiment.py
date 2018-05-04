from nltk.sentiment import *
from nltk.corpus import wordnet, sentiwordnet
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import pandas

"""
Function to calculate sentiment using three different lexicons
"""
def calculateSentiment(sentence):
    """
    calculate compound sentiment vector using three different lexicons, and separate sentiment values using VADER
    :param sentence: the sentence to be analysed
    :return: the compound vector with three values, one for each lexicon; and the separate value with three values: negative, neutral and positive
    """
    """
    Load in lexicons
    """
    slang = pandas.read_csv("SlangSD/SlangSD.txt", delimiter="\t", header=None)
    slang = slang.as_matrix()
    nltkSID = SentimentIntensityAnalyzer()

    """
    Get Vader compound score
    """
    polarityscore = nltkSID.polarity_scores(sentence)
    compound = list()
    compound.append(polarityscore["compound"])
    """
    Get Vader's negative, neutral and positive score
    """
    separate_sentiment = [polarityscore["neg"], polarityscore["neu"], polarityscore["pos"]]
    print("neg: ", polarityscore["neg"], "neutral: ", polarityscore["neu"], "pos: ", polarityscore["pos"])

    """
    Get compound score for SlangSD and sentiwordnet by looping over each word in the sentence
    """
    slangsentimentlist = list()
    score_list = []
    for word in word_tokenize(sentence):
        if word in slang[:,0]:
            index = np.where(slang[:,0] == word)
            slangsentimentlist.extend(slang[index,1].flatten())

        wnl = nltk.WordNetLemmatizer()

        tag = nltk.pos_tag([word])
        newtag = ''
        lemmatized = wnl.lemmatize(tag[0][0])
        if tag[0][1].startswith('NN'):
            newtag = 'n'
        elif tag[0][1].startswith('JJ'):
            newtag = 'a'
        elif tag[0][1].startswith('V'):
            newtag = 'v'
        elif tag[0][1].startswith('R'):
            newtag = 'r'
        else:
            newtag = ''
        synsets = list()
        if (newtag != ''):
            synsets = list(sentiwordnet.senti_synsets(lemmatized, newtag))
        # Getting average of all possible sentiments, as you requested
        score = 0
        if (len(synsets) > 0):
            for syn in synsets:
                score += syn.pos_score() - syn.neg_score()
            score_list.append(score / len(synsets))

    """
    Calculate final slangSD sentiment score
    """
    if len(slangsentimentlist) != 0:
        slangsentiment = sum([word_score for word_score in slangsentimentlist]) / len(slangsentimentlist)
    else:
        slangsentiment = 0
    compound.append(slangsentiment)
    """
    Calculate final sentiwordnet sentiment score
    """
    if len(score_list) != 0:
        sentence_sentiment = sum([word_score for word_score in score_list]) / len(score_list)
    else:
        sentence_sentiment = 0
    compound.append(sentence_sentiment)

    return compound, separate_sentiment









