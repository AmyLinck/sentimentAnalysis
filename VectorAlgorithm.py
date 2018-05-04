import numpy as np

"""
Calculate sentiment vectors
"""
def appendSentiment(sent_vector, word2vec):
  """
  Append separate sentiment vector to word2vec vector
  :param sent_vector: 
  :param word2vec: 
  :return: 
  """
  return np.append(word2vec, sent_vector)

def multiplyVectorWithSentiment(sent_vector, word2vec):
  """
  multiply compound sentiment vector with word2vec sentiment
  :param sent_vector: 
  :param word2vec: 
  :return: 
  """
  avg = sum([num for num in sent_vector]) /len(sent_vector)
  new_vector = []
  if avg != 0:
    new_vector.append(word2vec[0])
    for float in word2vec[1:]:
      new_vector.append(avg*float)
  else:
    new_vector = word2vec

  return new_vector

