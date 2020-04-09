
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

