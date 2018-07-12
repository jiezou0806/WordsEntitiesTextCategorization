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
# Query Entities in respect to Document Words.
def table_2(documents, entities, indexes):
    DTM_name = None
    documents_dict_name = None
    average_doc_length_name = None
    idf_words_name = None

    DTM_abstract = None
    documents_dict_abstract = None
    average_doc_length_abstract = None
    idf_words_abstract = None

    idf_entities_name = None
    idf_entities_abstract = None

    entities_names = entities.dropna(subset=['name'])
    entities_description = entities.dropna(subset=['abstract'])

    # Check if weights are already calculated and saved locally, to increase calculation speed.
    if os.path.isfile('datasets/table2/DTM_name.pic'):
        print('DTM name pickle found.')
        DTM_name = pd.read_pickle('datasets/table2/DTM_name.pic')
    else:
        (DTM_name, documents_dict_name, average_doc_length_name) = term_frequency(documents, entities_names, indexes, 1, 'name')

        DTM_name.to_pickle('./datasets/table2/DTM_name.pic')

    if os.path.isfile('datasets/table2/DTM_abstract.pic'):
        print('DTM abstract pickle found.')
        DTM_abstract = pd.read_pickle('datasets/table2/DTM_abstract.pic')
    else:
        (DTM_abstract, documents_dict_abstract, average_doc_length_abstract) = term_frequency(documents, entities_description, indexes, 1, 'abstract')

        DTM_abstract.to_pickle('./datasets/table2/DTM_abstract.pic')

    if os.path.isfile('datasets/table2/TFIDF_name.pic'):
        print('TFIDF name pickle found.')
        TFIDF_name = pd.read_pickle('datasets/table2/TFIDF_name.pic')
    else:
        if DTM_name is None or documents_dict_name is None or average_doc_length_name is None:
            (DTM_name, documents_dict_name, average_doc_length_name) = term_frequency(documents, entities_names, indexes, 1, 'name')

        idf_entities_name = idf(DTM_name)

        TFIDF_name = tf_idf(DTM_name, idf_entities_name, documents_dict_name)
        TFIDF_name.to_pickle('./datasets/table2/TFIDF_name.pic')

    if os.path.isfile('datasets/table2/TFIDF_abstract.pic'):
        print('TFIDF abstract pickle found.')
        TFIDF_abstract = pd.read_pickle('datasets/table2/TFIDF_abstract.pic')
    else:
        if DTM_abstract is None or documents_dict_abstract is None or average_doc_length_abstract is None:
            (DTM_abstract, documents_dict_abstract, average_doc_length_abstract) = term_frequency(documents, entities_description, indexes, 1, 'abstract')

        idf_entities_abstract = idf(DTM_abstract)

        TFIDF_abstract = tf_idf(DTM_abstract, idf_entities_abstract, documents_dict_name)
        TFIDF_abstract.to_pickle('./datasets/table2/TFIDF_abstract.pic')

    if os.path.isfile('datasets/table2/BM25_name.pic'):
        print('BM25 name pickle found.')
        BM25_name  = pd.read_pickle('datasets/table2/BM25_name.pic')
    else:
        if DTM_name is None or documents_dict_name is None or average_doc_length_name is None:
            (DTM_name, documents_dict_name, average_doc_length_name) = term_frequency(documents, entities_names, indexes, 1, 'name')

        if idf_entities_name is None:
            idf_entities_name = idf(DTM_name)

        BM25_name = bm25(DTM_name, idf_entities_name, documents_dict_name, average_doc_length_name)
        BM25_name.to_pickle('./datasets/table2/BM25_name.pic')

    if os.path.isfile('datasets/table2/BM25_abstract.pic'):
        print('BM25 abstract pickle found.')
        BM25_abstract = pd.read_pickle('datasets/table2/BM25_abstract.pic')
    else:
        if DTM_abstract is None or documents_dict_abstract is None or average_doc_length_abstract is None:
            (DTM_abstract, documents_dict_abstract, average_doc_length_abstract) = term_frequency(documents, entities_description, indexes, 1, 'abstract')

        if idf_entities_abstract is None:
            idf_entities_abstract = idf(DTM_abstract)

        BM25_abstract = bm25(DTM_abstract, idf_entities_abstract, documents_dict_abstract, average_doc_length_abstract)
        BM25_abstract.to_pickle('./datasets/table2/BM25_abstract.pic')

    if os.path.isfile('datasets/table2/BOOLEAN_name.pic'):
        print('BOOLEAN name pickle found.')
        BOOLEAN_name = pd.read_pickle('datasets/table2/BOOLEAN_name.pic')
    else:
        if DTM_name is None or documents_dict_name is None or average_doc_length_name is None:
            (DTM_name, documents_dict_name, average_doc_length_name) = term_frequency(documents, entities_names, indexes, 1, 'name')

        BOOLEAN_name = boolean(DTM_name, documents_dict_name)
        BOOLEAN_name.to_pickle('./datasets/table2/BOOLEAN_name.pic')

    if os.path.isfile('datasets/table2/BOOLEAN_abstract.pic'):
        print('BOOLEAN abstract pickle found.')
        BOOLEAN_abstract = pd.read_pickle('datasets/table2/BOOLEAN_abstract.pic')
    else:
        if DTM_abstract is None or documents_dict_abstract is None or average_doc_length_abstract is None:
            (DTM_abstract, documents_dict_abstract, average_doc_length_abstract) = term_frequency(documents, entities_description, indexes, 1, 'abstract')

        BOOLEAN_abstract = boolean(DTM_abstract, documents_dict_abstract)
        BOOLEAN_abstract.to_pickle('./datasets/table2/BOOLEAN_abstract.pic')

    if os.path.isfile('datasets/table2/LM_DS_name.pic'):
        print('LM_DS name pickle found.')
        LM_DS_name = pd.read_pickle('datasets/table2/LM_DS_name.pic')
    else:
        if DTM_name is None or documents_dict_name is None or average_doc_length_name is None:
            (DTM_name, documents_dict_name, average_doc_length_name) = term_frequency(documents, entities_names, indexes, 1, 'name')

        LM_name = lm(DTM_name, documents_dict_name)
        LM_DS_name = lm_ds(LM_name, DTM_name, documents_dict_name, average_doc_length_name)
        LM_DS_name.to_pickle('./datasets/table2/LM_DS_name.pic')

    if os.path.isfile('datasets/table2/LM_DS_abstract.pic'):
        print('LM_DS abstract pickle found.')
        LM_DS_abstract = pd.read_pickle('datasets/table2/LM_DS_abstract.pic')
    else:
        if DTM_abstract is None or documents_dict_abstract is None or average_doc_length_abstract is None:
            (DTM_abstract, documents_dict_abstract, average_doc_length_abstract) = term_frequency(documents, entities_description, indexes, 1, 'abstract')

        LM_abstract = lm(DTM_abstract, documents_dict_abstract)
        LM_DS_abstract = lm_ds(LM_abstract, DTM_abstract, documents_dict_abstract, average_doc_length_abstract)
        LM_DS_abstract.to_pickle('./datasets/table2/LM_DS_abstract.pic')

    TFIDF = TFIDF_name.join(TFIDF_abstract, lsuffix='_name', rsuffix='_abstract')
    BM25 = BM25_name.join(BM25_abstract, lsuffix='_name', rsuffix='_abstract')
    BOOLEAN = BOOLEAN_name.join(BOOLEAN_abstract, lsuffix='_name', rsuffix='_abstract')
    LM_DS = LM_DS_name.join(LM_DS_abstract, lsuffix='_name', rsuffix='_abstract')

    return BOOLEAN.join(TFIDF, lsuffix='_BOOLEAN', rsuffix='_TFIDF').join(BM25, lsuffix='_BOOLEAN', rsuffix='_BM25').join(LM_DS, rsuffix='_LM_DS')

# Function to calculate the term frequency weights for Table 2
def term_frequency(documents, entities, indexes, approach = 1, mode = 'name'):
    documents_dict = dict.fromkeys(indexes, None)
    vocabulary = []
    avg_length_docs = []

    print("Approach = " + str(approach))
    print("Entity mode = " + str(mode))

    for id in indexes:
        document = documents.loc[id]
        try:
            entities_id = entities[entities['doc_id'] == id]
            entities_names = " ".join(entities_id[mode])
            document_words = tokenize(document)
            avg_length_docs.append(len(document_words))
            entities_words = tokenize(entities_names)
            if approach == 1:
                document_words = ([x for x in entities_words if x in document_words])

        except Exception as e:
            print(e)

        vocabulary = vocabulary + document_words
        if approach == 1:
            documents_dict[id] = document_words
        else:
            documents_dict[id] = entities_words

    vocabulary = list(set(vocabulary))
    avg_doc_length = sum(avg_length_docs) / float(len(avg_length_docs))

    # Compute the Document Term Matrix
    print("Length of vocabulary documents words: " + str(len(vocabulary)))
    DTM = pd.DataFrame(0, columns=vocabulary, index=list(indexes))

    for document in documents_dict:
        try:
            for word in documents_dict[document]:
                DTM.at[document, word] += 1
        except Exception as e:
            print(e)

    # Only returns words which occur more than 3 times in the total of documents
    DTM = DTM[DTM.columns[DTM.sum(axis=0) > 3]]

    return (DTM, documents_dict, avg_doc_length)