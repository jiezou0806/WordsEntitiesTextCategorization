import pandas as pd
import os

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

def tagger(input_df):
    # Create lists for every feature
    tags = []

    feature_first_line = []
    feature_last_line = []
    feature_line_breaks = []

    feature_positive_words_header = []
    feature_positive_words_signature = []
    feature_number_words = []
    feature_line_ending = []

    content = []
    content_raw = []

    close = False

    for text in input_df['Description']:
        # Split question on double whitespaces, equals return in Trinicom
        lines = text.split('\n')
        lines = list(filter(None, lines))

        # Feature to define first line of inquiry, changes after first line is set
        first_line = True

        # Enumare over every line of text
        for i, line in enumerate(lines):
            line = line.strip()

            # if is empty, skip whole line in text
            if line == "":
                continue

            print(line)

            tag = input()
            if tag == 'q':
                close = True
                break
            elif tag == 's':
                break
            elif tag == '1':
                tags.append('header')
            elif tag == '2':
                tags.append('signature')
            elif tag == 'r':
                tags.pop()

                feature_first_line.pop()
                feature_last_line.pop()
                feature_line_breaks.pop()

                feature_positive_words_header.pop()
                feature_positive_words_signature.pop()
                feature_number_words.pop()
                feature_line_ending.pop()

                print(content.pop() + ' REMOVED')
                content_raw.pop()
                break
            else:
                tags.append('other')

            # Save original inquiry, lower the inquiry for feature extracting
            content_raw.append(line)
            line = str(line).lower()
            content.append(line)

            # Check if line is the first in inquiry
            if first_line == True:
                first_line = False
                feature_first_line.append(1)
            else:
                feature_first_line.append(0)

            # Check if line is last one in inquiry
            if i == len(lines) - 1:
                feature_last_line.append(1)
            else:
                feature_last_line.append(0)

            # Number of line breaks before this line
            feature_line_breaks.append(i)

            # Extract features from function
            positive_header, positive_signature, number_of_words, comma_ending = extract_features(line)

            feature_positive_words_header.append(int(positive_header))
            feature_positive_words_signature.append(int(positive_signature))
            feature_number_words.append(number_of_words)
            feature_line_ending.append(int(comma_ending))

        if close == True:
            break

    print(len(tags))
    print(len(content))
    print(len(content_raw))

    print(len(feature_first_line))
    print(len(feature_last_line))
    print(len(feature_line_breaks))

    print(len(feature_positive_words_header))
    print(len(feature_positive_words_signature))
    print(len(feature_number_words))
    print(len(feature_line_ending))

    tagger_df = pd.DataFrame(
        {'tag': tags, 'content': content, 'content_raw': content_raw, 'first_line': feature_first_line,
         'last_line': feature_last_line, 'line_breaks': feature_line_breaks,
         'positive_words_header': feature_positive_words_header,
         'positive_words_signature': feature_positive_words_signature, 'number_words': feature_number_words,
         'line_ending': feature_line_ending})
    return tagger_df

df = pd.read_pickle('original_dataset.pic')
tagger_df = tagger(df)
tagger_df.to_pickle('tagged_dataset.pic')


