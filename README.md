
# Set up the virtual environment

```bash
make virtualenv
source build/virtualenv/bin/activate
```
# Download the COVID data

Temporarily - download the 6GB of data from https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge and unzip the file in data/raw.

Future - sync the data from S3 by running:
```
make sync_data_from_s3
```
The version of the data in S3 was downloaded on 9th April 2020.

# Data

Research publications about COVID 19.
https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

- Commercial use subset (includes PMC content) -- 9118 full text (new: 128), 183Mb
- Non-commercial use subset (includes PMC content) -- 2353 full text (new: 385), 41Mb
- Custom license subset -- 16959 full text (new: 15533), 345Mb
- bioRxiv/medRxiv subset (pre-prints that are not peer reviewed) -- 885 full text (new: 110), 14Mb
- Metadata file -- 60Mb

# Initial thoughts on the dataset and models

### Dataset

It was generally nice to work with the well structured JSONs and a metadata file, however there were some sticking points:

- Not so obvious why the `pdf_json` and `pmc_json` were in different folders? Why was it important to separate them and could you not provide a tag within the json to say if the json came from pdf or pmc or both (if it is important to include this info)?
- We wanted to link the text from the publications to the correct metadata to get the `publish_time` attribute, and also we needed to remove duplication of publications from the `pdf_json` datasources and the `pmc_json` datasources. Because of the unique identifier not being included in the jsons it meant we had to query the metadata using either PMCID or the sha ID (depending on whether the data was from the `pdf_json` or `pmc_json` datasource). Would be good to include the cord ID in the json. Also would be good to have a unique ID that spread over both the pdf and the pmc jsons so that we could quickly remove duplicates.
- Why were chunks of the body text broken up and what does section refer to here (section 245?) ? e.g.
```
"body_text": [{ "text": "The underlying cause ...vered 294 from the filter types in each sample had an effect on viral RNA copy variability.", "cite_spans": [], "ref_spans": [], "section": "245" }, { "text": "After washing the filt... not peer-reviewed) is the author/funder. It . https://doi.org/10.1101/441154 doi: bioRxiv preprint", "cite_spans": [], "ref_spans": [], "section": "295" }]
```

### Models

We explored a couple of different Q&A models for the task. We settle on a pipeline consisting of:
1. Tf-idf vectorizer to mine the relevant articles to a certain question
2. Scibert to re-rank the most relevant ones.
3. Sentence tokenizer in the most relevant ones with scispacy.
4. Bert/Scibert to find the answer spans in the text.

After providing answer spans, we annotated, a posteriori, some dataset to find out whether the answers were relevant or not and evaluate models.
It is unclear whether Step 2 actually increased the relevant metrics, but using Step 3 with scispacy clearly produced better results over naive sentence tokenizers.

### Other thoughts

Given the effect that fine tuning has on QA datasets, it would be great to have some annotated questions and answers to fine tune BERT like models to make them more effective. Additionally it would be nice if there was some ground truth in the challenge, for example few annotated documents that were relevant to questions that could be used to evaluate different strategies.

Ideally we would like to have some of the above datasets so that we could train a BERT reranking model on top of an information retrieval algorithm as well as fine tune BERT for QA. One question we had was to which extent BERT can be used to answer questions directly without the need of information retrieval, understandably this might computationally expensive but maybe there are ways it could become feasible.
