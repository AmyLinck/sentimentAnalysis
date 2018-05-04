from Word2Vec import *
from Preprocess import *
from ClusterAlgorithm import *
from Sentiment import *
from VectorAlgorithm import *
import pandas

#'dbow+dmm'
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


"""
Read original data
"""
data = pandas.read_csv("Data/master.csv")
sentences = data["text"]

"""
Pre-process sentences
"""
processed = [process(sentence, "nltk") for sentence in sentences[:]]
#pandas.DataFrame(processed).to_csv("Data/processed_sentences_short.csv",header=False) #Write fo file

"""
Word2Vec sentence representation
"""
vectors, model = train_word2vec(processed)
#pandas.DataFrame(vectors).to_csv("Data/simple_vectors_short.csv",header=False) #Write fo file
vectors = np.asarray(vectors)
#sentences = pandas.read_csv("Data/processed_sentences_short.csv", header=None, encoding="ISO-8859-1") #Read in processed sentences from file

"""
Sentiment vector calculation
"""
compound = []
seperate = []
sentences = sentences.as_matrix()
for sent in sentences:
    if not is_number(sent[1]):
        compound_part, seperate_part = calculateSentiment(sent[1])
        print(compound_part)
        print(seperate_part)
        compound.append(compound_part)
        seperate.append(seperate_part)
    else:
        compound.append([0])
        seperate.append([0])
#pandas.DataFrame(compound).to_csv("Data/compound_sentiment_vectors_short.csv",header=False) #Write fo file
#pandas.DataFrame(seperate).to_csv("Data/neg_neu_pos_sentiment_vectors_short.csv",header=False) #Write fo file

"""
Sentiment addition and multiplication vectors
"""
added = []
multiply = []
for i,sent in enumerate(seperate):
    added.append(appendSentiment(seperate[i,1:], vectors[i,:]))
    multiply.append(multiplyVectorWithSentiment(compound[i,1:], vectors[i,:]))
added = np.asarray(added)
print(added)
added = np.nan_to_num(added)
multiply = np.asarray(multiply)
multiply = np.nan_to_num(multiply)


"""
Clustering
"""
#DBS(vectors[:,:], sentences[1],0.025, 5)
KMC(vectors[:,:], sentences[1], 2)
