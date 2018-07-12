from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
import os
import pandas as pd
import numpy as np

def extract_features(input):
    # Cast to string and strip whitespaces
    input = str(input).strip()

    # Lower input
    line = input.lower()

    # Check if line contains greatings
    header_words = ['geacht', 'geacht', 'beste', 'goedemorgen', 'goede morgen', 'ls', 'l.s.' 'hallo', 'hoi', 'hi', 'dag']
    positive_header = False

    for word in header_words:
        if word in line:
            positive_header = True
            break

    # Check if input contains endings
    ending_words = ['groet', 'vriendelijk', 'regards', 'mvg', 'm.v.g.']
    positive_ending = False

    for word in ending_words:
        if word in line:
            positive_ending = True
            break

    number_of_words = len(line.split())

    # Check if input ends with a comma
    comma_ending = False
    try:
        if line[-1] == ',':
            comma_ending = True
    except:
        comma_ending = False

    return positive_header, positive_ending, number_of_words, comma_ending

openings = []
closings = []

def extract_content(text, model):
    lines = text.split('\n')

    output = []
    first_line = True
    found_signature = False

    for line_break, line in enumerate(lines):
        if found_signature:
            break

        line = line.strip()
        if line == '':
            continue

        if first_line == True:
            first_line = False

        if line_break == len(lines) - 1:
            last_line = True
        else:
            last_line = False

        positive_header, positive_ending, number_of_words, comma_ending = extract_features(line)

        pred = model.predict([[int(first_line), int(last_line), line_break, comma_ending, number_of_words,
                               positive_header, positive_ending]])

        # print(line)

        if pred == [1]:
            openings.append(line)
            output = []
        elif pred == [2]:
            closings.append(line)
            found_signature = True
            break
        else:
            output.append(line)

    return ' '.join(output)

tagged = shuffle(pd.read_pickle('dataset_tagged.pic'))

X = []
y = []

h = 0
s = 0
o = 0

for index, row in tagged.iterrows():
    if row['tag'] == 'header':
        h+=1
        if h > 150:
            continue

    if row['tag'] == 'signature':
        s += 1
        if s > 150:
            continue

    if row['tag'] == 'other':
        o += 1
        if o > 150:
            continue

    X.append([row['first_line'], row['last_line'], row['line_breaks'], row['line_ending'], row['number_words'],
              row['positive_words_header'], row['positive_words_signature']])

    if row['tag'] == 'header':
        y.append(1)
    elif row['tag'] == 'signature':
        y.append(2)
    else:
        y.append(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

kernels = ('linear', 'poly', 'rbf')
for kernel in kernels:
    print()
    print(kernel + ':')
    clf = svm.SVC(kernel=kernel, C=1)
    score = cross_val_score(clf, X_train, y_train, cv=5)
    print('Cross validation scores: ' + str(score.mean()))

clf = svm.SVC(kernel='poly', C=1)
clf.fit(X, y)

df = pd.read_pickle('regex_dataset.pic')
df['Description'] = df['Description'].apply(lambda x: extract_content(x, clf))
df['Description'] = df['Description'].str.strip()
df['Description'].replace('', np.nan, inplace=True)
df = df.dropna(subset=['Description'])
df.to_pickle('cleaned_dataset.pic'))
