from transformers import BertTokenizer, BertForQuestionAnswering

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import torch

import pickle
import json
import time
import os
from argparse import ArgumentParser

class QuestionCovid:

    def __init__(
            self,
            TOKENIZER,
            MODEL,
            index2paperID,
            index2paperPath
            ):
        self.TOKENIZER = TOKENIZER
        self.MODEL = MODEL
        self.index2paperID = index2paperID
        self.index2paperPath = index2paperPath

    def fit(self, data_text):

        self.TFIDF_VECTORIZER = TfidfVectorizer()
        print("   Fitting TFIDF", end=" ", flush=True)
        start = time.time()
        self.TFIDF_VECTORIZER.fit(data_text.values())
        print(" - Took {:.2f}s".format(time.time()-start))
        print("   Creating Articles matrix", end="", flush=True)
        start = time.time()
        self.ARTICLES_MATRIX = self.TFIDF_VECTORIZER.transform(data_text.values())
        print(" - Took {:.2f}s".format(time.time()-start))

    def get_answer(self, text, question):

        input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
        input_ids = self.TOKENIZER.encode(input_text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = self.MODEL(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = self.TOKENIZER.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

        return answer, start_scores.max(), end_scores.max()

    def predict(self, question):

        query = self.TFIDF_VECTORIZER.transform([question])
        best_matches = sorted([(i,c) for i, c in enumerate(cosine_similarity(query, self.ARTICLES_MATRIX).ravel())], key=lambda x: x[1], reverse=True)
        best_score = 0 # if score is negative, i consider the answer wrong
        best_answer = None
        best_text = None
        for i, _ in best_matches[:5]:
            print(self.index2paperID[i])
            paper_path = self.index2paperPath[i]
            with open(paper_path) as json_file:
                article_data = json.load(json_file)
                text = ' '.join([d['text'] for d in article_data['body_text']])

            for text_i in range(0, len(text), 512):
                subtext = text[text_i:text_i+512]
                answer, start_score, end_score = self.get_answer(subtext, question)
                if start_score > best_score:
                    best_score = start_score
                    best_answer = answer
                    best_text = text
        if best_score:
            return [best_text, best_answer]
        else:
            return "No answer"

def get_data_texts(articles_dir, articles_folders):

        data_text = {}
        index2paperID = {}
        index2paperPath = {}
        i = 0
        for articles_folder in articles_folders:
            json_files = os.listdir(articles_dir + articles_folder)
            for json_file in json_files:
                paper_path = articles_dir + articles_folder + json_file 
                with open(paper_path) as json_file:
                    article_data = json.load(json_file)
                    data_text[article_data['paper_id']] = ' '.join([d['text'] for d in article_data['body_text']])
                    index2paperID[i] = article_data['paper_id']
                    index2paperPath[i] = paper_path
                    i += 1

        return data_text, index2paperID, index2paperPath

def train():
    print("   Loading BERT")
    TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
    MODEL = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    articles_dir = 'data/raw/CORD-19-research-challenge/'
    articles_folders = [
        'biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/',
        'comm_use_subset/comm_use_subset/pdf_json/',
        'comm_use_subset/comm_use_subset/pmc_json/',
        'noncomm_use_subset/noncomm_use_subset/pdf_json/',
        'noncomm_use_subset/noncomm_use_subset/pmc_json/',
        'custom_license/custom_license/pdf_json/',
        'custom_license/custom_license/pmc_json/']

    print("   Ingesting publications", end=" ", flush=True)
    start = time.time()
    data_text, index2paperID, index2paperPath = get_data_texts(articles_dir, articles_folders)
    print("- Took {:.2f}s".format(time.time()-start))

    covid_q = QuestionCovid(TOKENIZER, MODEL, index2paperID, index2paperPath)
    covid_q.fit(data_text)
    return covid_q

def create_argparser():

    parser = ArgumentParser()
    parser.add_argument(
        '--question',
        help='Type in a question',
        default='What is the incubation period?'
    )
    parser.add_argument(
        '--pre_trained_model_path',
        help='Path to pretrained covid q model'
    )

    return parser

if __name__ == '__main__':

    parser = create_argparser()
    args = parser.parse_args()

    if args.pre_trained_model_path:
        print("Loading pre trained model")
        with open(args.pre_trained_model_path, "rb") as f:
            covid_q = pickle.load(f)
    else:
        print("Training new model")
        covid_q = train()
        print("   Saving model")
        with open("models/covid_q.pkl", "wb") as f:
            pickle.dump(covid_q, f)

    print("Finding best answer")
    start = time.time()
    _, best_answer = covid_q.predict(args.question)

    print("----- Answer: -----")
    print(best_answer, end=" ", flush=True)
    print(" - Took {}s".format(time.time()-start))
