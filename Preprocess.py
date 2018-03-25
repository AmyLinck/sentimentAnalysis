from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, sentiment
from nltk.corpus import wordnet, sentiwordnet
import numpy as np
import re
from wordcloud import WordCloud,STOPWORDS

def cleanText(text, lemmatize, stemmer):
    if isinstance(text,float):
        text = str(text)
    soup = BeautifulSoup(text,"lxml")
    text = soup.get_text()
    text = re.sub(r"(#|@)[^\s]*", " ", text)
    #text = text.lower()

    if lemmatize:
        wordnet_lemmatizer = WordNetLemmatizer()

        def get_tag(tag):
            if tag.startswith('J'):
                return wordnet.ADJ
            elif tag.startswith('V'):
                return wordnet.VERB
            elif tag.startswith('N'):
                return wordnet.NOUN
            elif tag.startswith('R'):
                return wordnet.ADV
            else:
                return ''

        text_result = []
        tokens = word_tokenize(text)  # Generate list of tokens
        tagged = pos_tag(tokens)
        for t in tagged:
            try:
                text_result.append(wordnet_lemmatizer.lemmatize(t[0], get_tag(t[1][:2])))
            except:
                text_result.append(wordnet_lemmatizer.lemmatize(t[0]))
        return text_result

    if stemmer:
        text_result = []
        tokens = word_tokenize(text)
        snowball_stemmer = SnowballStemmer('english')
        for t in tokens:
            text_result.append(snowball_stemmer.stem(t))
        return text_result

    if not stemmer and not lemmatize:
            return text


def calculateEmphasis(text):
    for t in text:
        emph = re.findall(r'[^\w\s,]', t)

def calculateSentiment(text):
    nltkSID = sentiment.SentimentIntensityAnalyzer()
    nltkSentiment = list()
    for t in text:
        polarityscore = nltkSID.polarity_scores(t)
        nltkSentiment.extend(polarityscore)
