import spotlight
import pandas as pd
from tqdm import tqdm
import os

host = 'http://localhost:2222/rest/annotate'


def retrieve_entities(text):
    annotations = spotlight.annotate(
        host,
        text,
        confidence=0,
        support=0,
        spotter='Default',
    )
    return annotations


def compute_dataset(df, column):
    print('Dataset length before: ' + str(len(df)))
    entities_documents_df = pd.DataFrame(columns=['doc_id', 'original', 'URI'])

    for index, row in tqdm(df.iterrows()):
        try:
            entities_text = retrieve_entities(row[column])

            for annotation in entities_text:
                entities_documents_df = entities_documents_df.append({
                                                            'doc_id': index,
                                                            'URI': annotation['URI'],
                                                            'original': annotation['surfaceForm']},
                                                            ignore_index=True
                )
        except Exception as e:
            print(e)
            print(row[column])
            continue

    print('Dataset length after: ' + str(len(entities_documents_df['doc_id'].unique())))
    print('Amount of entities found: ' + str(len(entities_documents_df)))

    return entities_documents_df

df = pd.read_pickle('cleaned_dataset.pic')
df_annotated = compute_dataset(df, 'Description')
df_annotated.to_pickle(os.path.join(salesforce_path, 'spotlight/cleaned_annotated_dataset.pic'))