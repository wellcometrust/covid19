from transformers import BertTokenizer, BertForQuestionAnswering, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wasabi import msg, table, row
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
        self.nlp = pipeline("question-answering")

    def fit(self, data_text):

        self.TFIDF_VECTORIZER = TfidfVectorizer()
        with msg.loading("   Fitting TFIDF"):
            start = time.time()
            self.TFIDF_VECTORIZER.fit(data_text.values())
        msg.good("   TFIDF fitted - Took {:.2f}s".format(time.time()-start))
        with msg.loading("   Creating Articles matrix"):
            start = time.time()
            self.ARTICLES_MATRIX = self.TFIDF_VECTORIZER.transform(data_text.values())
        msg.good("   Article matrix created - Took {:.2f}s".format(time.time()-start))

    def get_answer(self, text, question):

        input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
        input_ids = self.TOKENIZER.encode(input_text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = self.MODEL(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = self.TOKENIZER.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
        score = round(start_scores.max().item(), 2)
        #response = self.nlp({'question': question, 'context': text})
        #answer = response["answer"]
        #score = response["score"]

        return answer, score

    def predict(self, question):

        query = self.TFIDF_VECTORIZER.transform([question])
        best_matches = sorted([(i,c) for i, c in enumerate(cosine_similarity(query, self.ARTICLES_MATRIX).ravel())], key=lambda x: x[1], reverse=True)

        for i, _ in best_matches[:5]:
            best_score = 0 # if score is negative, i consider the answer wrong
            best_answer = None
            best_text = None
            
            paper_path = self.index2paperPath[i]
            with open(paper_path) as json_file:
                article_data = json.load(json_file)
                text = ' '.join([d['text'] for d in article_data['body_text']])

            for text_i in range(0, len(text), 512):
                subtext = text[text_i:text_i+512]
                answer, score = self.get_answer(subtext, question)
                if score > best_score:
                    best_score = score
                    best_answer = answer
                    best_text = subtext
            yield (self.index2paperID[i], best_answer, best_score, best_text)

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
    with msg.loading("   Loading BERT"):
        TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
        MODEL = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    msg.good("   BERT loaded")

    articles_dir = 'data/raw/CORD-19-research-challenge/'
    articles_folders = [
        'biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/',
        'comm_use_subset/comm_use_subset/pdf_json/',
        'comm_use_subset/comm_use_subset/pmc_json/',
        'noncomm_use_subset/noncomm_use_subset/pdf_json/',
        'noncomm_use_subset/noncomm_use_subset/pmc_json/',
        'custom_license/custom_license/pdf_json/',
        'custom_license/custom_license/pmc_json/']

    with msg.loading("   Loading publications"):
        start = time.time()
        data_text, index2paperID, index2paperPath = get_data_texts(articles_dir, articles_folders)
    msg.good("   Publications loaded - Took {:.2f}s".format(time.time()-start))

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
    parser.add_argument(
        '--answers_path',
        help='Path to save answers',
        default='answers.jsonl'
    )

    return parser

if __name__ == '__main__':

    parser = create_argparser()
    args = parser.parse_args()

    if args.pre_trained_model_path:
        with msg.loading("Loading pre trained model"):
            with open(args.pre_trained_model_path, "rb") as f:
                covid_q = pickle.load(f)
        msg.good(f"Loaded {args.pre_trained_model_path}")
    else:
        msg.text("Training new model")
        covid_q = train()
        msg.good("Trained")
        with open("models/covid_q.pkl", "wb") as f:
            pickle.dump(covid_q, f)


    print("Finding best answer")
    start = time.time()
    with open(args.answers_path, "w") as f:
        header = ["Paper id", "Answer", "Score", "Snippet"]
        widths = (40, 40, 5, 35)
        print(table((), header=header, divider=True, widths=widths))
        for paper_id, answer, score, snippet in covid_q.predict(args.question):
            data = (paper_id, answer, score, snippet[:15] + '...' + snippet[-15:])
            print(row(data, widths=widths))
            chunk = json.dumps({
                'paper_id': paper_id,
                'answer': answer,
                'snippet': snippet,
                'score': score
            })
            f.write(chunk + '\n')


        # save to file for inspection

    print("\nTook {}s".format(time.time()-start))
