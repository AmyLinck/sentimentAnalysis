import gensim.models.doc2vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from collections import OrderedDict
import multiprocessing

"""
Calculate word2vec vectors using the paragraph version of word2vec, which adds an extra input/output (depending on the method used) to represent the sentence vector
"""
def train_word2vec(data):
    """
    calculate vectors using gensim
    :param data:
    :return:
    """

    """
    add token to each line to id each sentence
    """
    docs = []
    for line_no, line in enumerate(data):
        tokens = gensim.utils.to_unicode(line).split()
        words = tokens[1:]
        tags = [line_no]
        docs.append(TaggedDocument(words, tags))

    """
    paramaters
    """
    alpha, min_alpha, passes = (0.025, 0.001, 20)
    alpha_delta = (alpha - min_alpha) / passes

    """
    threading for speed
    """
    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1

    """
    Calculate the three different possible  sentence2vec models
    """
    simple_models = [
        # DM w/ concatenation - window=5 (both sides) approximates paper's 10-word total window size
        Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
        # DBOW
        Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
        # DM w/ average
        Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
    ]

    simple_models[0].build_vocab(docs)  # PV-DM w/ concat requires one special NULL word so it serves as template
    for model in simple_models[1:]:
        model.reset_from(simple_models[0])

    """
    concate dbow and dmm models for actual model to be used
    """
    models_by_name = OrderedDict((str(model), model) for model in simple_models)
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    train_model = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

    """
    Train the resulting model
    """
    train_model.alpha, train_model.min_alpha = alpha, min_alpha
    train_model.train(data, total_examples=len(data), epochs=3)
    vectors = []
    for i in range(len(data)):
        vectors.append(train_model.docvecs[i])
        #model.infer_vector(data[id])

    """
    return the vectors and the trained model
    """
    return vectors, train_model



