# Import standard packages
import pandas as pd
import os

# Use metrics and tokenizer from functions files
from functions.tokenizer import tokenize
from functions.metrics.idf import idf
from functions.metrics.tfidf import tf_idf
from functions.metrics.boolean import boolean
from functions.metrics.bm25 import bm25
from functions.metrics.lm_ds import lm, lm_ds
from functions.lda import LDA_table

# Functions to extract all features for Table 1.
# Query Words in respect to Document Words.
def table_1(documents, indexes):
    documents_dict = None
    average_doc_length = None
    idf_words = None

    # Check if weights are already calculated and saved locally, to increase calculation speed.
    if os.path.isfile('datasets/table1/DTM.pic'):
        print('DTM pickle found.')
        DTM = pd.read_pickle('datasets/table1/DTM.pic')
    else:
        print('Creating DTM')
        (DTM, documents_dict, average_doc_length) = term_frequency(documents, indexes)
        DTM.to_pickle('./datasets/table1/DTM.pic')

    if os.path.isfile('datasets/table1/TFIDF.pic'):
        print('TFIDF pickle found.')
        TFIDF = pd.read_pickle('datasets/table1/TFIDF.pic')
    else:
        print('Creating TFIDF.')
        if DTM is None or documents_dict is None or average_doc_length is None:
            (DTM, documents_dict, average_doc_length) = term_frequency(documents, indexes)

        idf_words = idf(DTM)
        TFIDF = tf_idf(DTM, idf_words, documents_dict)
        TFIDF.to_pickle('./datasets/table1/TFIDF.pic')

    if os.path.isfile('datasets/table1/BOOLEAN.pic'):
        print('BOOLEAN pickle found.')
        BOOLEAN = pd.read_pickle('datasets/table1/BOOLEAN.pic')
    else:
        print('Creating BOOLEAN.')
        if DTM is None or documents_dict is None or average_doc_length is None:
            (DTM, documents_dict, average_doc_length) = term_frequency(documents, indexes)

        BOOLEAN = boolean(DTM, documents_dict)
        BOOLEAN.to_pickle('./datasets/table1/BOOLEAN.pic')

    if os.path.isfile('datasets/table1/LM.pic'):
        print('LM pickle found.')
        LM = pd.read_pickle('datasets/table1/LM.pic')
    else:
        print('Creating Language Model.')
        if DTM is None or documents_dict is None or average_doc_length is None:
            (DTM, documents_dict, average_doc_length) = term_frequency(documents, indexes)

        LM = lm(DTM, documents_dict)
        LM.to_pickle('./datasets/table1/LM.pic')

    if os.path.isfile('datasets/table1/LM_DS.pic'):
        print('LM_DS pickle found.')
        LM_DS = pd.read_pickle('datasets/table1/LM_DS.pic')
    else:
        print('Creating LM_DS.')
        if DTM is None or documents_dict is None or average_doc_length is None:
            (DTM, documents_dict, average_doc_length) = term_frequency(documents, indexes)

        LM_DS = lm_ds(LM, DTM, documents_dict, average_doc_length)
        LM_DS.to_pickle('./datasets/table1/LM_DS.pic')

    if os.path.isfile('datasets/table1/BM25.pic'):
        print('BM25 pickle found.')
        BM25 = pd.read_pickle('datasets/table1/BM25.pic')
    else:
        print('Creating BM25.')
        if DTM is None or documents_dict is None or average_doc_length is None:
            (DTM, documents_dict, average_doc_length) = term_frequency(documents, indexes)

        if idf_words is None:
            idf_words = idf(DTM)

        BM25 = bm25(DTM, idf_words, documents_dict, average_doc_length)
        BM25.to_pickle('./datasets/table1/BM25.pic')

    if os.path.isfile('datasets/table1/LDA.pic'):
        print('DTM pickle found.')
        LDA = LDA_table(DTM)
    else:
        print('Creating LDA')
        (DTM, documents_dict, average_doc_length) = term_frequency(documents, indexes)
        LDA = LDA_table(DTM)
        LDA.to_pickle('./datasets/table1/LDA.pic')

    return BOOLEAN.join(TFIDF, lsuffix='_BOOLEAN', rsuffix='_TFIDF').join(BM25, lsuffix='_BOOLEAN', rsuffix='_BM25').join(LM_DS, rsuffix='_LM_DS').join(LDA)

# Function to calculate the term frequency weights for Table 1
def term_frequency(documents, indexes):
    documents_dict = dict.fromkeys(indexes, None)
    vocabulary = []
    average_length_docs = []

    for id in indexes:
        document = documents.loc[id]
        try:
            words = tokenize(document)
            average_length_docs.append(len(words))
            vocabulary = vocabulary + words
            documents_dict[id] = words
        except Exception as e:
            print(e)

    vocabulary = list(set(vocabulary))
    average_doc_length = sum(average_length_docs) / float(len(average_length_docs))

    # Compute the Document Term Matrix
    DTM = pd.DataFrame(0, columns=vocabulary, index=list(indexes))

    for document in documents_dict:
        try:
            for word in documents_dict[document]:
                DTM.at[document, word] += 1
        except Exception as e:
            print(e)

    # Only returns words which occur more than 3 times in the total of documents
    DTM = DTM[DTM.columns[DTM.sum(axis=0) > 3]]
    return (DTM, documents_dict, average_doc_length)



