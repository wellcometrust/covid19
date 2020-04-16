from transformers import BertTokenizer, BertForQuestionAnswering, BertModel, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wasabi import msg, table, row
import pandas as pd
import torch
import spacy
from langdetect import detect

import pickle
import json
import time
import os
from argparse import ArgumentParser

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
SCIBERT_PATH = os.path.join(SCRIPT_PATH, "../models/scibert/scibert_scivocab_uncased")

from download_scibert import download_scibert

download_scibert()

class QuestionCovid:

    def __init__(
            self,
            TOKENIZER,
            MODEL,
            index2paperID,
            index2paperPath,
            use_bert_qa=True,
            ):
        self.TOKENIZER = TOKENIZER
        self.MODEL = MODEL
        self.scibert_tokenizer = BertTokenizer.from_pretrained(SCIBERT_PATH)
        self.scibert_model = BertModel.from_pretrained(SCIBERT_PATH, output_hidden_states=True)

        self.index2paperID = index2paperID
        self.index2paperPath = index2paperPath

        self.nlp = pipeline("question-answering")
        self.scispacy = spacy.load("en_core_sci_sm")

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

        return answer, score

    def rank_shortlisted(self, question, list_of_texts):
        """ Ranks question/document pairs based on scibert"""

        # Load scibert

        embedding_question = self._scibert_embedding(question)

        similarities = [(i,
                         tf_idf_score,
                         cosine_similarity(embedding_question, self._scibert_embedding(text))[0, 0],
                         text)
                        for i, tf_idf_score, text in list_of_texts]
        similarities = sorted(similarities, key=lambda x: x[2], reverse=True)

        return similarities

    def _scibert_embedding(self, x):
        # Max sequence length is 512 for BERT
        if len(x) > 512:
            embedded_a = self._scibert_embedding(x[:512])
            embedded_b = self._scibert_embedding(x[512:])
            return embedded_a + embedded_b

        tokenized_x = self.scibert_tokenizer.tokenize("[CLS] " + x + " [SEP]")
        indexed_tokens = self.scibert_tokenizer.convert_tokens_to_ids(tokenized_x)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.zeros(tokens_tensor.shape, dtype=torch.long)

        with torch.no_grad():
            output = self.scibert_model(tokens_tensor, token_type_ids=segments_tensor)

        embedded_x = torch.stack(output[2][-4:]).mean(dim=0).mean(dim=1)

        return embedded_x.cpu().numpy().flatten().reshape(1, -1)

    def predict(self, question, n_tf_idf_matches=20, n_scibert_matches=5):

        query = self.TFIDF_VECTORIZER.transform([question + ' covid'])
        best_matches = sorted([(i, c) for i, c in enumerate(cosine_similarity(query, self.ARTICLES_MATRIX).ravel())], key=lambda x: x[1], reverse=True)

        best_match_texts = []

        # Opens the paper file and gets the text
        for i, tfidf_score in best_matches[:n_tf_idf_matches]:
            paper_path = self.index2paperPath[i]
            with open(paper_path) as json_file:
                article_data = json.load(json_file)
                text = ' '.join([d['text'] for d in article_data['body_text']])
            best_match_texts += [(i, tfidf_score, text)]

        # Re-ranks everything from Scibert
        best_match_texts = self.rank_shortlisted(list_of_texts=best_match_texts,
                                                 question=question)

        for i, tfidf_score, scibert_score, text in best_match_texts[:n_scibert_matches]:
            best_score = 0  # if score is negative, i consider the answer wrong
            best_answer = "No answer"
            best_text = "No snippet"

            n = 3

            sentences = [s.text for s in self.scispacy(text).sents]
            
            def yield_subtext(sentences):
                subtext = ''
                for sent in sentences:
                    if len(sent) + len(subtext) > 450:
                        yield subtext
                        subtext = sent
                    else:
                        subtext += sent

            for subtext in yield_subtext(sentences):
                answer, score = self.get_answer(subtext, question)
                if score > best_score:
                    best_score = score
                    best_answer = answer
                    best_text = subtext
            yield (self.index2paperID[i], best_answer, best_score, best_text, tfidf_score, scibert_score)

def get_data_texts(articles_dir, articles_folders, meta_path):

        def get_abstract_language(abstract):
            try:
                language = detect(abstract)
            except:
                language = None
            return language

        # Create dict of paper_id and publication year
        meta_data = pd.read_csv(meta_path, low_memory=True)
        paperID2year = {}
        paperID2lang = {}
        sha2pmcid = {}
        for _, meta_row in meta_data.iterrows():
            # Only save information for meta data with parsed text
            if meta_row['has_pmc_xml_parse'] or meta_row['has_pdf_parse']:
                # The paper ID will either be the pmcid or sha
                if pd.notnull(meta_row['pmcid']):
                    paperID2year[meta_row['pmcid']] = meta_row['publish_time']
                    if pd.notnull(meta_row['abstract']):
                        lang = get_abstract_language(meta_row['abstract'])
                        if lang:
                            paperID2lang[meta_row['pmcid']] = lang
                # There can be muliple sha IDs in the rows
                if pd.notnull(meta_row['sha']):
                    lang = None
                    if pd.notnull(meta_row['abstract']):
                        lang = get_abstract_language(meta_row['abstract'])
                    paper_ids = meta_row['sha'].split('; ')
                    for paper_id in paper_ids:
                        if pd.notnull(meta_row['pmcid']):
                            sha2pmcid[paper_id] = meta_row['pmcid']
                        paperID2year[paper_id] = meta_row['publish_time']
                        if lang:
                            paperID2lang[paper_id] = lang

        data_text = {}
        index2paperID = {}
        index2paperPath = {}
        paperpmcids = set()
        i = 0
        for articles_folder in articles_folders:
            json_files = os.listdir(articles_dir + articles_folder)
            for json_file in json_files:
                paper_path = articles_dir + articles_folder + json_file 
                with open(paper_path) as json_file:
                    article_data = json.load(json_file)
                    # Don't include duplicates (defined from pmcid - if given) in data_text
                    if article_data['paper_id'][0:3] == 'PMC':
                        pmcid = article_data['paper_id']
                    else:
                        pmcid = sha2pmcid.get(article_data['paper_id'], None)
                    if (not pmcid) or (pmcid not in paperpmcids):
                        if pmcid:
                            paperpmcids.add(pmcid)
                        paper_date = paperID2year.get(article_data['paper_id'], None)
                        paper_language = paperID2lang.get(article_data['paper_id'], None)
                        if paper_date:
                            # Only include papers from 2020 and papers in English (or no language given)
                            if (paper_date[0:4] == '2020') and (paper_language == 'en' or not paper_language):
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

    articles_dir = os.path.join(SCRIPT_PATH, '../data/raw/CORD-19-research-challenge/')
    articles_folders = [
        'biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/',
        'comm_use_subset/comm_use_subset/pdf_json/',
        'comm_use_subset/comm_use_subset/pmc_json/',
        'noncomm_use_subset/noncomm_use_subset/pdf_json/',
        'noncomm_use_subset/noncomm_use_subset/pmc_json/',
        'custom_license/custom_license/pdf_json/',
        'custom_license/custom_license/pmc_json/']
    meta_path = articles_dir + 'metadata.csv'

    with msg.loading("   Loading publications"):
        start = time.time()
        data_text, index2paperID, index2paperPath = get_data_texts(articles_dir, articles_folders, meta_path)
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


    challenge_tasks = [
      {
          "task": "What is known about transmission, incubation, and environmental stability?",
          "questions": [
              "Is the virus transmitted by aerosol, droplets, food, close contact, fecal matter, or water?",
              "How long is the incubation period for the virus?",
              "Can the virus be transmitted asymptomatically or during the incubation period?",
              "How does weather, heat, and humidity affect the tramsmission of 2019-nCoV?",
              "How long can the 2019-nCoV virus remain viable on common surfaces?"
          ]
      },
      {
          "task": "What do we know about COVID-19 risk factors?",
          "questions": [
              "What risk factors contribute to the severity of 2019-nCoV?",
              "How does hypertension affect patients?",
              "How does heart disease affect patients?",
              "How does copd affect patients?",
              "How does smoking affect patients?",
              "How does pregnancy affect patients?",
              "What is the fatality rate of 2019-nCoV?",
              "What public health policies prevent or control the spread of 2019-nCoV?"
          ]
      },
      {
          "task": "What do we know about virus genetics, origin, and evolution?",
          "questions": [
              "Can animals transmit 2019-nCoV?",
              "What animal did 2019-nCoV come from?",
              "What real-time genomic tracking tools exist?",
              "What geographic variations are there in the genome of 2019-nCoV?",
              "What effors are being done in asia to prevent further outbreaks?"
          ]
      },
      {
          "task": "What do we know about vaccines and therapeutics?",
          "questions": [
              "What drugs or therapies are being investigated?",
              "Are anti-inflammatory drugs recommended?"
          ]
      },
      {
          "task": "What do we know about non-pharmaceutical interventions?",
          "questions": [
              "Which non-pharmaceutical interventions limit tramsission?",
              "What are most important barriers to compliance?"
          ]
      },
      {
          "task": "What has been published about medical care?",
          "questions": [
              "How does extracorporeal membrane oxygenation affect 2019-nCoV patients?",
              "What telemedicine and cybercare methods are most effective?",
              "How is artificial intelligence being used in real time health delivery?",
              "What adjunctive or supportive methods can help patients?"
          ]
      },
      {
          "task": "What do we know about diagnostics and surveillance?",
          "questions": [
              "What diagnostic tests (tools) exist or are being developed to detect 2019-nCoV?"
          ]
      },
      {
          "task": "Other interesting questions",
          "questions": [
              "What is the immune system response to 2019-nCoV?",
              "Can personal protective equipment prevent the transmission of 2019-nCoV?",
              "Can 2019-nCoV infect patients a second time?"
          ]
      }
    ]
    with open(args.answers_path, "w") as f:
        for task_id, task in enumerate(challenge_tasks):
            task_question = task['task']
            msg.text(f"Task {task_id}: {task_question}")

            questions = task['questions']
            for question_id, question in enumerate(questions):
                with msg.loading(f"Answering question: {question}"):
                    start = time.time()
                    for paper_id, answer, score, snippet, tfidf_score, scibert_score in covid_q.predict(question):
                        chunk = json.dumps({
                            'task_id': task_id,
                            'task': task_question,
                            'question_id': question_id,
                            'question': question,
                            'paper_id': paper_id,
                            'answer': answer,
                            'snippet': snippet,
                            'bert_score': score,
                            'tfidf_score': tfidf_score
                        })
                        f.write(chunk + '\n')
                time_elapsed = time.time()-start
                msg.good(f"Question {question_id} answered - Took {time_elapsed}s")
