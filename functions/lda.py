import pandas as pd
import scipy.sparse
import lda

def LDA_table(DTM):
    model = lda.LDA(n_topics=32, n_iter=2000, random_state=1)
    result = model.fit_transform(scipy.sparse.csr_matrix(DTM.values))

    df_result = pd.DataFrame(result, DTM.index)

    return df_result

'''import numpy as np
from tqdm import tqdm
import pandas as pd
from functions.tokenizer import tokenize
from scipy.sparse import coo_matrix


df = pd.read_excel('../datasets/websitecasecsj.xlsx')
df = df.dropna(subset=['Category', 'Subcategory', 'Description'])

print(df.columns)

docs = {}

for index, row in tqdm(df.iterrows()):
    try:
        docs[index] = tokenize(row['Description'])
    except Exception as e:
        print(e)
        continue

n_nonzero = 0
vocab = set()
for docterms in docs.values():
    unique_terms = set(docterms)    # all unique terms of this doc
    vocab |= unique_terms           # set union: add unique terms of this doc
    n_nonzero += len(unique_terms)  # add count of unique terms in this doc

# make a list of document names
# the order will be the same as in the dict
docnames = list(docs.keys())

#unique terms
print(len(vocab))
print(n_nonzero)

docnames = np.array(docnames)
vocab = np.array(list(vocab))

vocab_sorter = np.argsort(vocab)    # indices that sort "vocab"

ndocs = len(docnames)
nvocab = len(vocab)

data = np.empty(n_nonzero, dtype=np.intc)     # all non-zero term frequencies at data[k]
rows = np.empty(n_nonzero, dtype=np.intc)     # row index for kth data item (kth term freq.)
cols = np.empty(n_nonzero, dtype=np.intc)     # column index for kth data item (kth term freq.)

ind = 0     # current index in the sparse matrix data
# go through all documents with their terms
for docname, terms in docs.items():
    # find indices into  such that, if the corresponding elements in  were
    # inserted before the indices, the order of  would be preserved
    # -> array of indices of  in
    term_indices = vocab_sorter[np.searchsorted(vocab, terms, sorter=vocab_sorter)]

    # count the unique terms of the document and get their vocabulary indices
    uniq_indices, counts = np.unique(term_indices, return_counts=True)
    n_vals = len(uniq_indices)  # = number of unique terms
    ind_end = ind + n_vals  #  to  is the slice that we will fill with data

    data[ind:ind_end] = counts                  # save the counts (term frequencies)
    cols[ind:ind_end] = uniq_indices            # save the column index: index in
    doc_idx = np.where(docnames == docname)     # get the document index for the document name
    rows[ind:ind_end] = np.repeat(doc_idx, n_vals)  # save it as repeated value

    ind = ind_end  # resume with next document -> add data to the end

dtm = coo_matrix((data, (rows, cols)), shape=(ndocs, nvocab), dtype=np.intc)

import lda
# do 1000 LDA is default
model = lda.LDA(n_topics=50, n_iter=500, random_state=1)

model.fit_transform(dtm)


print(model.topic_word_)
ndz = model.ndz_
for doc in ndz:
    print(doc)
    break

for doc in model.doc_topic_:
    print(doc)
    break'''