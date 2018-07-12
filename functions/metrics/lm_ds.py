# Function compute the Language Model weight using term frequencies.
import pandas as pd

def lm(DTM, documents_dict):
    LM = pd.DataFrame(0, columns=DTM.columns, index=list(DTM.index))

    for document in documents_dict:
        document_length = len(documents_dict[document])

        for word in documents_dict[document]:
            try:
                LM.at[document, word] = DTM.at[document, word] / document_length
            except Exception as e:
                print(e)

    return LM

# Function to compute for every feature the  Language Model with Dirichlet Smoothing weight using original LM matrix.
def lm_ds(LM, DTM, documents_dict, avg_doc_length):
    LM_DS = pd.DataFrame(0, columns=DTM.columns, index=list(DTM.index))
    u = 3

    collection_lm = {}
    for word, values in DTM.iteritems():
        collection_lm[word] = sum(values) / avg_doc_length * len(documents_dict)

    for document in documents_dict:
        document_length = len(documents_dict[document])

        for word in documents_dict[document]:
            try:
                LM_DS.at[document, word] = (document_length / (document_length + u)) * LM.at[document, word] + ((u / (u + document_length)) * collection_lm[word])
            except Exception as e:
                print(e)

    return LM_DS