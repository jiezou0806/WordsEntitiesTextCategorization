import pandas as pd
import gc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

# Load Dataframes, including dataframe with spotted entities of documents
df = pd.read_pickle('original_dataset.pic')
df_entities = pd.read_pickle('entities_dataset_per_document.pic')

# Compute all tables
table1 = table_1(df['Description'], df.index.values)
table2 = table_2(df['Description'], df_entities, df.index.values)
table3 = table_3(df['Description'], df_entities, df.index.values)
table4 = table_4(df_entities, df.index.values)
gc.collect()

# Join all tables for TC
result = table1.join(table2, lsuffix='_T1', rsuffix='_T2').join(table3, rsuffix='_T3').join(table4, rsuffix='_T4').join(labels)

# Dump garbage to prevent Memory Errors
gc.collect()

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
    model_name = model.__class__.__name__
    print(model)
    accuracies = cross_val_score(model, result.loc[:, result.columns != 'Label column name'], result['Label column name'], scoring='accuracy', cv=CV)

    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

print(cv_df.groupby('model_name').accuracy.mean())