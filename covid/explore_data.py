from collections import Counter
import json
import os

import pandas as pd

articles_dir = 'data/raw/CORD-19-research-challenge/'
articles_folders = [
    'biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/',
    'comm_use_subset/comm_use_subset/pdf_json/',
    'comm_use_subset/comm_use_subset/pmc_json/',
    'noncomm_use_subset/noncomm_use_subset/pdf_json/',
    'noncomm_use_subset/noncomm_use_subset/pmc_json/',
    'custom_license/custom_license/pdf_json/',
    'custom_license/custom_license/pmc_json/'
]

def produce_metatada(articles_folders):
    i = 0
    data = []
    for articles_folder in articles_folders:
        json_files = os.listdir(articles_dir + articles_folder)
        for json_file in json_files:
            if i % 1000 == 0:
                print(i)

            paper_path = articles_dir + articles_folder + json_file 
            with open(paper_path) as f:
                article_data = json.load(f)

            article_title = article_data['metadata']['title']
            paper_id = article_data['paper_id']
            article_text = ' '.join([d['text'] for d in article_data['body_text']])
            data.append({
                'paper_id': paper_id,
                'paper_filename': json_file[:-5],
                'paper_title': article_title,
                'is covid': any([k in article_text for k in ['covid', 'sars', 'mers', 'corona']])
            })
            i += 1
    return pd.DataFrame(data)

if not os.path.exists('data/processed/pub_metadata.csv'):
    pub_metadata = produce_metatada(articles_folders)
    pub_metadata.to_csv('data/processed/pub_metadata.csv')
else:
    pub_metadata = pd.read_csv('data/processed/pub_metadata.csv')

metadata = pd.read_csv("data/raw/CORD-19-research-challenge/metadata.csv")

# Missing metadata
publications_sha_list = [f for f in pub_metadata['paper_filename'] if 'PMC' not in f]
publications_pmcid_list = [f[:-4] for f in pub_metadata['paper_filename'] if 'PMC' in f]
metadata_sha_list = [s.strip() for m in metadata['sha'] if not pd.isna(m) for s in m.split(';')]
metadata_pmc_list = [m for m in metadata['pmcid'] if not pd.isna(m)]
print("Missing metadata in sha {} publications.".format(len(set(publications_sha_list) - set(metadata_sha_list))))
print("Missing metadata in pmc {} publications.".format(len(set(publications_pmcid_list) - set(metadata_pmc_list))))

# Duplicate sha
def yield_duplicates(metadata_list):
    for m, c in Counter(metadata_list).most_common():
        if c > 1:
            yield (m, c)
with open('data/processed/duplicates.txt', 'w') as f:
    nb_dubs = 0
    for m, c in yield_duplicates(metadata_sha_list):
        f.write("{m} {c}")
        nb_dubs += 1
    for m, c in yield_duplicates(metadata_pmc_list):
        f.write("{m} {c}")
        nb_dubs += 1
print("Duplicate publications in metadata: {}. Complete list at data/processed/duplicates.txt".format(nb_dubs))
nb_dubs = len(list(yield_duplicates(pub_metadata['paper_id']))) + len(list(yield_duplicates(pub_metadata['paper_filename'])))
print("Duplicate publications in files: {}".format(nb_dubs))

covid_pubs = pub_metadata['is covid'].sum()
print("There {} covid related pubs out of {}".format(covid_pubs, len(pub_metadata)))


#processed_metadata = metadata.merge(pub_metadata, how='left', left_on='sha', right_on='paper_id')
#print(processed_metadata[(processed_metadata['title'].str.lower() != processed_metadata['paper_title'].str.lower()) & processed_metadata['paper_title']][['title', 'paper_title']])

