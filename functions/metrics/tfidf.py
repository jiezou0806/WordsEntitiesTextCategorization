# Function to compute a tf-idf.py matrix based on the term frequency matrix and the idf.py per word
import pandas as pd

def tf_idf(DTM, idf, documents_dict):
    TFIDF = pd.DataFrame(0, columns=DTM.columns, index=list(DTM.index))
    for document in documents_dict:
        try:
            for word in documents_dict[document]:
                try:
                    TFIDF.at[document, word] += idf[word]
                except Exception as e:
                    print(e)

        except Exception as e:
            print(e)

    return TFIDF