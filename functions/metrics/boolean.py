# Function to calculate boolean weight based on term frequencies
import pandas as pd

def boolean(DTM, documents_dict):
    BOOLEAN = pd.DataFrame(0, columns=DTM.columns, index=list(DTM.index))
    for document in documents_dict:
        try:
            for word in documents_dict[document]:
                if word in DTM.columns:
                    BOOLEAN.at[document, word] = 1
        except Exception as e:
            print(e)

    return BOOLEAN