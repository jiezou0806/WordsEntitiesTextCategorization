# Function to calculate inverse document frequency for tf-idf.py
import math

def idf(DTM):
    total_docs = len(DTM)
    idf = {}

    for feature in DTM:
        idf[feature] = math.log(total_docs / sum(DTM[feature] > 0))

    return idf