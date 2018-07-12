# Function to compute BM-25 matrix
import pandas as pd

def bm25(DTM, idf, documents_dict, average_doc_length):
    BM25 = pd.DataFrame(0, columns=DTM.columns, index=list(DTM.index))

    k_1 = 1.6
    b = 0.75

    for document in documents_dict:
        document_length_divided_avg = len(documents_dict[document]) / average_doc_length
        try:
            for word in documents_dict[document]:
                tf_word = DTM.at[document, word]
                BM25.at[document, word] = idf[word] * ((tf_word * (k_1 + 1)) / (tf_word + k_1 * (1 - b + b * document_length_divided_avg)))
        except Exception as e:
            print(e)

    return BM25
