from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation

import nltk.data
import re

stemmer = nltk.stem.SnowballStemmer('dutch')
wnl = nltk.stem.WordNetLemmatizer()
stop = stopwords.words('dutch')
'''
stop = stop +     ['albert', 
                   'heijn',  
                   'klantenservic',
                   'ahold.com', 
                   'e-mail', 
                   'nasa-nr', 
                   'contactpersoon',
                   'groep',
                   'vriendelijk',
                   'groet',
                  'ah',
                  'ah.nl']
'''


def tokenize(text):
    tokens_ = [word_tokenize(sent) for sent in sent_tokenize(text, 'dutch')]

    tokens = []
    for token_by_sent in tokens_:
        tokens += token_by_sent

    tokens = list(map(lambda t: stemmer.stem(t), tokens))
    tokens = list(filter(lambda t: t.lower() not in stop, tokens))
    tokens = list(filter(lambda t: t not in punctuation, tokens))

    filtered_tokens = []
    for token in tokens:
        #token = wnl.lemmatize(token)
        if re.search('[a-zA-z]', token):
            filtered_tokens.append(token)

    filtered_tokens = list(map(lambda token: token.lower(), filtered_tokens))
    # stemmed_tokens = list(map(lambda token: stemmer.stem(token), filtered_tokens))
    return filtered_tokens