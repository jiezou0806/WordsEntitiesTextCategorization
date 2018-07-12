import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

from functions.table3 import table_3

# Load dataframes, including dataframe with spotted entities of documents
df = pd.read_pickle('original_dataset.pic')
df_entities = pd.read_pickle('entities_dataset_per_document.pic')

# Use a Pandas Dataframe, one columns as input for documents, other parameters for index of df
table3 = table_3(df['Description'], df_entities, df.index.values)

# Join corresponding labels of documents to the computed weights table of Table 1:
result = table3.join(labels)

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
