from pathlib import Path
import time
import json
import os

from langdetect import detect as detect_language
from wasabi import msg
import pandas as pd


SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
ARTICLES_DIR = os.path.join(SCRIPT_PATH, '../data/raw/CORD-19-research-challenge/')


def get_language(text):
    try:
        language = detect_language(text)
    except:
        language = "unknown"
    return language

def yield_data(articles_dir):
    meta_path = articles_dir + 'metadata.csv'
    metadata = pd.read_csv(meta_path, low_memory=False)

    metadata.drop_duplicates(subset=['pmcid'], inplace=True)
    metadata.drop_duplicates(subset=['sha'], inplace=True)

    metadata['publish_date'] = pd.to_datetime(metadata['publish_time'])
    metadata = metadata[metadata['publish_date'].dt.year == 2020]

    metadata['lang'] = metadata['abstract'].apply(lambda x: get_language(x))
    metadata = metadata[metadata['lang'] == 'en']

    valid_ids = set([sha.strip() for shas in metadata['sha'] for sha in shas.split(';')])
    valid_ids.update(set(metadata['pmcid']))

    for dirname, _, filenames in os.walk(articles_dir):
        for filename in filenames:
            article_path = os.path.join(dirname, filename)
            if article_path[-4:] != 'json':
                continue
            with open(article_path) as json_file:
                article_data = json.load(json_file)
            article_id = article_data['paper_id']
            if article_id not in valid_ids:
                continue
            article_text = ' '.join([d['text'] for d in article_data['body_text']])
            yield article_id, article_text


if __name__ == '__main__':
    with msg.loading("Processing articles"):
        start = time.time()

        output_dir = os.path.join(SCRIPT_PATH,"../data/processed/publications/")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for article_id, article_text in yield_data(ARTICLES_DIR):
            with open(os.path.join(output_dir, f"{article_id}.txt"), "w") as f:
                f.write(article_text)
    msg.good("Articles processed - Took {:.2f}s".format(time.time()-start))

