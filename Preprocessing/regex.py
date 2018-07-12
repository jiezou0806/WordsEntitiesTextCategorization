import re
import pandas as pd
import os

from functions.tokenizer import tokenize

def do_regex(input, column=None):
    expr = {
        # email
        r'\S*@\S*\s?': ' ',
        # IP
        r'[0-9]+(?:\.[0-9]+){3}': ' ',
        # url
        r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*': ' ',
        # date
        r'\d+[\/:\-]\d+[\/:\-\s]*[\dAaPpMm]*': ' ',
        r'\w+\s\d+[\,]\s\d+': ' ',
        # bankrekening nummer
        r'[a-zA-Z]{2}[0-9]{2}[a-zA-Z0-9]{4}[0-9]{7}([a-zA-Z0-9]?){0,16}': ' ',
        # remove all prices
        r'[0-9]*[,.]+([0-9]{1,2})?': ' ',
    }

    if isinstance(input, str):
        output = input
        for key in expr:
            output = re.sub(key, expr[key], output)

        return output
    elif isinstance(input, pd.DataFrame):
        print("Input is Pandas DataFrame.")
        if column:
            print('Is column.')
            try:
                input[column].replace(expr, regex=True, inplace=True)
            except Exception as e:
                print(e)
        return input

    else:
        print('No correct input type.')

df = pd.read_pickle('original_dataset.pic')
df = do_regex(df, column='Description')
df.to_pickle('regex_datset.pic')