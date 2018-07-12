import pandas as pd
from tqdm import tqdm
import os
from SPARQLWrapper import SPARQLWrapper, JSON, XML, N3, RDF
from time import sleep

sparql = SPARQLWrapper("http://nl.dbpedia.org/sparql")
sparql.setReturnFormat(JSON)

def return_dbpedia(URI):
    sparql.setQuery(("""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        SELECT ?abstract ?name
        WHERE {
            <%s> dbo:abstract ?abstract.
            <%s> rdfs:label ?name
        }
    """) % (URI, URI))
    try:
        results = sparql.query().convert()
    except Exception as e:
        print(e)
        print(URI)
        results = None

    try:
        name = results["results"]["bindings"][0]['name']['value']
    except:
        name = None

    try:
        abstract = results["results"]["bindings"][0]['abstract']['value']
    except:
        abstract = None

    return [name, abstract]


def query_ents(df):
    print('Dataset length before: ' + str(len(df)))
    ents = df['URI'].unique()
    # print(ents)
    ents_dbpedia_df = pd.DataFrame(columns=['URI', 'name', 'abstract'])

    for URI in tqdm(ents):
        sleep(0.25)
        try:
            name, abstract = return_dbpedia(URI)
            ents_dbpedia_df = ents_dbpedia_df.append({'URI': URI, 'name': name, 'abstract': abstract},
                                                     ignore_index=True)
        except Exception as e:
            print(URI)
            print(e)
            continue

    print('Dataset length after: ' + str(len(ents_dbpedia_df)))
    return ents_dbpedia_df


def link_docs_ents(df, name, path):
    dbpedia_results = query_ents(df)

    df = df.merge(dbpedia_results, on=['URI'], how='left')
    df.to_pickle(os.path.join(path, name + '.pic'))

    return df

df = pd.read_pickle('cleaned_annotated_dataset.pic')

link_docs_ents(df, 'entities_dataset', dataset_path)