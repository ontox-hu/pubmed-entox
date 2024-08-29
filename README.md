# Pubmed-entox: extracting biomedical relationships from the PubMed database.

This repo contains an example of run an NLP pipeline on the PubMed biobrick to extract relationships between chemicals and biomedical phenotypes.

* pubmed_run.py runs through the entire example, from loading the brick in a Spark session to outputting a parquet dataframe containing and relationships found.

* requirements.txt details the required environment to run this script.

This works relies heavily on [Biobricks](https://biobricks.ai/) and on the [en-tox](https://github.com/ontox-project/en-tox) NLP model.

