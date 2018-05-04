from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from twokenize import *
from nltk.corpus import wordnet, sentiwordnet
import numpy as np
import re, pandas, grammar_check, autocorrect
from wordcloud import WordCloud,STOPWORDS

def process(text, postagger):
    """
    Pre-processing of the sentence: identify language, delete hashtags and @ as well as webpagelinks, correct spelling and grammar
    :param text:
    :param postagger:
    :return:
    """
    tagged = []
    emo = []
    tool = grammar_check.LanguageTool('en-GB')
    if postagger =="nltk":
        if isinstance(text, float):
            text = str(text)
        text = re.sub(r"http\S+", "", text) #Remove URLs
        text = re.sub("RT", "", text) #remove retweet tag
        soup = BeautifulSoup(text, "lxml")
        text = soup.get_text()
        text = re.sub(r"(#|@)[^\s]*", "", text) #Removes hashtags and @name
        text = text.replace("|", "")
        text = text.replace(" da ", " the ")
        text = text.replace(" ya ", " you ")
        text = text.replace(" u ", " you ")
        #emo = re.findall(r'[^\w\s,]', text) #Find emoticons and special characters and save in list
        matches = tool.check(text)
        print(matches)
        text = grammar_check.correct(text,  matches)
        tokens = word_tokenize(text)  # Generate list of tokens
        tagged = pos_tag(tokens) #POS tag words
        spell_text = []
        """
        Correct spelling as well as possible
        """
        for word in tagged:
            pattern = re.compile(r"[^a-zA-Z\d\s]")
            #word[1] != ":" and word[1] != "." and word[1] != "," and word[1] != ")" and word[1] != "("
            if word[1] != "CD" and not pattern.match(word[1]) and word[1] != "CC" and word[1] != "POS":
                word_point = word[0].replace(".", "")
                corrected = autocorrect.spell(word_point)
            else :
                corrected = word[0]
            spell_text.append(corrected)
        text = ' '.join(word for word in spell_text)


    return text