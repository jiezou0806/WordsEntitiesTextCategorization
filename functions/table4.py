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

# Functions to extract all features for Table 1.
# Query Entities in respect to Document Entities.
def table_4( entities, indexes):
    ETM_soft = None
    entities_dict_soft = None
    avg_doc_length_soft = None
    idf_soft = None

    ETM_hard = None
    entities_dict_hard = None
    avg_doc_length_hard = None
    idf_hard = None

    LM_soft = None
    LM_hard = None

    # First check if table parts already are computed and saved to pickle to enhance speed
    if os.path.isfile('datasets/table4/ETM_soft.pic'):
        print('ETM soft pickle found.')

        ETM_soft = pd.read_pickle('datasets/table4/ETM_soft.pic')
    else:
        ETM_soft, entities_dict_soft, avg_doc_length_soft = term_frequency(entities.dropna(subset=['URI']), indexes, 'URI')

        ETM_soft.to_pickle('./datasets/table4/ETM_soft.pic')

    # First check if table parts already are computed and saved to pickle to enhance speed
    if os.path.isfile('datasets/table4/ETM_hard.pic'):
        print('ETM soft pickle found.')

        ETM_hard = pd.read_pickle('datasets/table4/ETM_hard.pic')
    else:
        ETM_hard, entities_dict_hard, avg_doc_length_hard = term_frequency(entities.dropna(subset=['original']), indexes, 'original')

        ETM_hard.to_pickle('./datasets/table4/ETM_hard.pic')

    if os.path.isfile('datasets/table4/TFIDF_soft.pic'):
        print('TFIDF soft pickle found.')

        TFIDF_soft = pd.read_pickle('datasets/table4/TFIDF_soft.pic')
    else:
        if ETM_soft is None or entities_dict_soft is None or avg_doc_length_soft is None:
            ETM_soft, entities_dict_soft, avg_doc_length_soft = term_frequency(entities.dropna(subset=['URI']), indexes, 'URI')

        idf_soft = idf(ETM_soft)
        TFIDF_soft = tf_idf(ETM_soft, idf_soft, entities_dict_soft)
        ETM_soft.to_pickle('./datasets/table4/TFIDF_soft.pic')

    if os.path.isfile('datasets/table4/TFIDF_hard.pic'):
        print('TFIDF hard pickle found.')
        TFIDF_hard = pd.read_pickle('datasets/table4/TFIDF_hard.pic')
    else:
        if ETM_hard is None or entities_dict_hard is None or avg_doc_length_hard is None:
            ETM_hard, entities_dict_hard, avg_doc_length_hard = term_frequency(entities.dropna(subset=['original']), indexes, 'original')

        idf_hard = idf(ETM_hard)
        TFIDF_hard = tf_idf(ETM_hard, idf_hard, entities_dict_hard)
        TFIDF_hard.to_pickle('./datasets/table4/TFIDF_hard.pic')

    if os.path.isfile('datasets/table4/BM25_soft.pic'):
        print('BM25 soft pickle found.')
        BM25_soft = pd.read_pickle('datasets/table4/BM25_soft.pic')
    else:
        if ETM_soft is None or entities_dict_soft is None or avg_doc_length_soft is None:
            ETM_soft, entities_dict_soft, avg_doc_length_soft = term_frequency(entities.dropna(subset=['URI']), indexes, 'URI')

        if idf_soft is None:
            idf_soft = idf(ETM_soft)

        BM25_soft = bm25(ETM_soft, idf_soft, entities_dict_soft, avg_doc_length_soft)
        BM25_soft.to_pickle('./datasets/table4/BM25_soft.pic')

    if os.path.isfile('datasets/table4/BM25_hard.pic'):
        print('BM25 hard pickle found.')
        BM25_hard = pd.read_pickle('datasets/table4/BM25_hard.pic')
    else:
        if ETM_hard is None or entities_dict_hard is None or avg_doc_length_hard is None:
            ETM_hard, entities_dict_hard, avg_doc_length_hard = term_frequency(entities.dropna(subset=['original']), indexes, 'original')

        if idf_hard is None:
            idf_hard = idf(ETM_hard)

        BM25_hard = bm25(ETM_hard, idf_hard, entities_dict_hard, avg_doc_length_hard)
        BM25_hard.to_pickle('./datasets/table4/BM25_hard.pic')

    if os.path.isfile('datasets/table4/BOOLEAN_soft.pic'):
        print('BOOLEAN soft pickle found.')
        BOOLEAN_soft = pd.read_pickle('datasets/table4/BOOLEAN_soft.pic')
    else:
        if ETM_soft is None or entities_dict_soft is None or avg_doc_length_soft is None:
            ETM_soft, entities_dict_soft, avg_doc_length_soft = term_frequency(entities.dropna(subset=['URI']), indexes, 'URI')

        BOOLEAN_soft = boolean(ETM_soft, entities_dict_soft)
        BOOLEAN_soft.to_pickle('./datasets/table4/BOOLEAN_soft.pic')

    if os.path.isfile('datasets/table4/BOOLEAN_hard.pic'):
        print('BOOLEAN hard pickle found.')
        BOOLEAN_hard = pd.read_pickle('datasets/table4/BOOLEAN_hard.pic')
    else:
        if ETM_hard is None or entities_dict_hard is None or avg_doc_length_hard is None:
            ETM_hard, entities_dict_hard, avg_doc_length_hard = term_frequency(entities.dropna(subset=['original']), indexes, 'original')

        BOOLEAN_hard = boolean(ETM_hard, entities_dict_hard)
        BOOLEAN_hard.to_pickle('./datasets/table4/BOOLEAN_hard.pic')

    if os.path.isfile('datasets/table4/LM_DS_soft.pic'):
        print('LM soft pickle found.')
        LM_DS_soft = pd.read_pickle('datasets/table4/LM_DS_soft.pic')
    else:
        if ETM_soft is None or entities_dict_soft is None or avg_doc_length_soft is None:
            ETM_soft, entities_dict_soft, avg_doc_length_soft = term_frequency(entities.dropna(subset=['URI']), indexes, 'URI')

        if LM_soft is None:
            LM_soft = lm(ETM_soft, entities_dict_soft)

        LM_DS_soft = lm_ds(LM_soft, ETM_soft, entities_dict_soft, avg_doc_length_soft)
        LM_DS_soft.to_pickle('./datasets/table4/LM_DS_soft.pic')

    if os.path.isfile('datasets/table4/LM_DS_hard.pic'):
        print('LM soft pickle found.')
        LM_DS_hard = pd.read_pickle('datasets/table4/LM_DS_hard.pic')
    else:
        if ETM_hard is None or entities_dict_hard is None or avg_doc_length_hard is None:
            ETM_hard, entities_dict_hard, avg_doc_length_hard = term_frequency(entities.dropna(subset=['original']), indexes, 'original')

        if LM_hard is None:
            LM_hard = lm(ETM_hard, entities_dict_hard)

        LM_DS_hard = lm_ds(LM_hard, ETM_hard, entities_dict_hard, avg_doc_length_hard)
        LM_DS_hard.to_pickle('./datasets/table4/LM_DS_hard.pic')

    TFIDF =  TFIDF_soft.join(TFIDF_hard, lsuffix='_soft', rsuffix='_hard')
    LM_DS = LM_DS_soft.join(LM_DS_hard, lsuffix='_soft', rsuffix='_hard')
    BM25 = BM25_soft.join(BM25_hard, lsuffix='_soft', rsuffix='_hard')
    BOOLEAN = BOOLEAN_soft.join(BOOLEAN_hard, lsuffix='_soft', rsuffix='_hard')

    return BOOLEAN.join(TFIDF, lsuffix='_BOOLEAN', rsuffix='_TFIDF').join(BM25, lsuffix='_BOOLEAN', rsuffix='_BM25').join(LM_DS, rsuffix='_LM_DS')

# Function to calculate the term frequency weights for Table 4
def term_frequency(entities, indexes, mode='name'):
    entities_dict = dict.fromkeys(indexes, None)
    vocabulary = []
    avg_length_docs = []

    print("Entity mode = " + str(mode))

    for id in indexes:
        try:
            entities_id = entities[entities['doc_id'] == id]
            entities_names = " ".join(entities_id[mode])
            entities_words = tokenize(entities_names)
            avg_length_docs.append(len(entities_words))
        except Exception as e:
            print(e)
            print(id)

        vocabulary = vocabulary + entities_words

        entities_dict[id] = entities_words

    vocabulary = list(set(vocabulary))

    avg_doc_length = sum(avg_length_docs) / float(len(avg_length_docs))

    print("Length of vocabulary documents words: " + str(len(vocabulary)))

    DEM = pd.DataFrame(0, columns=vocabulary, index=list(indexes))

    for document in entities_dict:
        try:
            for entity in entities_dict[document]:
                DEM.at[document, entity] += 1
        except Exception as e:
            print(e)
            print(entity)

    DEM = DEM[DEM.columns[DEM.sum(axis=0) > 3]]

    return (DEM, entities_dict, avg_doc_length)





