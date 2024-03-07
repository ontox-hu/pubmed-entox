import biobricks as bb
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructType, StructField, IntegerType, ArrayType
import spacy
from spacy.matcher import DependencyMatcher
from typing import List
import os
import sys

#os.environ['PYSPARK_PYTHON'] = sys.executable
#os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder\
    .appName("entox run on pubmed")\
    .config("spark.executor.memory", "4g")\
    .config("spark.driver.memory", "2g")\
    .getOrCreate()
    
pubmed = bb.assets('pubmed').pubmed_parquet
df = spark.read.parquet(pubmed)

# 2. BUILD A TITLE ABSTRACT DATAFRAME ==============================================

# UDF to extract full abstract and title from JSON
#TODO: also extract author list and year?
def build_abstracttext(json_string):
    try:
        parsed_json = json.loads(json_string)
        title = parsed_json["MedlineCitation"]["Article"].get("ArticleTitle", "")
        abstract = parsed_json["MedlineCitation"]["Article"].get("Abstract",{}).get("AbstractText", "")
        #'\n'.join(abstract) if isinstance(abstract, list) else \
        #   '\n'.join(abstract.values()) if isinstance(abstract, dict) else abstract
        if isinstance(abstract, list) and '@Label' in abstract[0]:
            # De-nest abstract text further 
            abstract = " ".join([item["#text"] for item in abstract])
        # Make sure to return a string, even if the title is missing
        return f"{title}", f"{abstract}"
    except Exception as e:
        # Log the error for debugging
        print("Error:", e)
        return "Error", str(e)

# Selecting a small sample for testing
sample_df = df.limit(100)       
sample_df.show(10)
# showing specific cell value (row i col j) without collecting entire pyspark df
# sample_sample_df.limit(i+1).collect()[i][j]

#TODO: fix abstracts where parsing is different

# Register UDF with Spark
build_abstract_udf = F.udf(build_abstracttext, returnType=StructType([
        StructField("title", StringType()),
        StructField("abstract", StringType())
    ]))

# Apply the UDF to create new columns
sample_df = sample_df.withColumn("parsed_columns", build_abstract_udf(F.col("json")))
sample_df = sample_df.select(
    F.col("PMID"),
    F.col("parsed_columns.title").alias("title"),
    F.col("parsed_columns.abstract").alias("abstract")
    )
sample_df.show(20)

# 3. RUN NLP ENTOX MODEL ON ALL ABSTRACTS TEXT ==============================================

# test df for when biobricks is not accessible
#test_df = spark.read.option("header",True).csv("pubmed_run/test.csv")
 
# load spacy model
nlp = spacy.load("en_tox")
nlp.add_pipe("merge_entities")

def dependency_matcher(nlp, ent_cause, ent_effect): 
    '''
    Input:
    - nlp (English): A spacy model with customized ner that can recognize entities of interest
    - ent_cause: a string corresponding to the first entity type of interest (cause) as used in nlp model (ex: "COMPOUND"). 
    - ent_effect: a string corresponding to the second entity type of interest (effect) as used in nlp model (ex: "PHENOTYPE"). 
    Output:
    - matcher: DependencyMatcher with the pattern 'LINKING_VERB' added
    '''
    matcher = DependencyMatcher(nlp.vocab)
    
    # Create pattern that explains the relation: a verb is an (indirect) ancestor of a phenotype and a compound
    '''
    Every dictionary contains one node in the dependency tree
    RIGHT_ID describes the node (e.g. the verb, the cause or the effect)
    RIGHT_ATTRS describes how the model can recognize this node
    LEFT_ID describes where the node is linked to (the cause and the effect are linked to the verb
    REL_OP describes the relation between the two nodes ('>>' is indirect child)
    '''
    pattern = [
        {'RIGHT_ID': 'linking_verb',
         'RIGHT_ATTRS': {'POS':'VERB'}
        },
        {'LEFT_ID':'linking_verb',
         "REL_OP": ">>",
         "RIGHT_ID": "cause",
         "RIGHT_ATTRS": {"ENT_TYPE": ent_cause}
        },
        {"LEFT_ID": "linking_verb",
         "REL_OP": ">>",
         "RIGHT_ID": "effect",
         "RIGHT_ATTRS": {"ENT_TYPE": ent_effect}
        }
    ] 
    # Add the pattern to the matcher and call it "LINKING_VERB"
    matcher.add("LINKING_VERB", [pattern])
    return matcher

matcher = dependency_matcher(nlp, "COMPOUND", "PHENOTYPE")
causal_verbs = ['increase', 'produce', 'cause', 'induce', 'generate', 'effect', 'provoke', 'arouse', 'elicit', 'lead', 'trigger','derive', 'associate', 'relate', 'link', 'stem', 'originate', 'lead', 'bring', 'result', 'inhibit', 'elevate', 'diminish', "exacerbate", "decrease"]

# start with sample_df from last step 2

# create UDF 
# TODO: udf for each function?? Does it make it faster?
def extract_sentences(doc) -> List[str]:  
    sentences = [sent.text for sent in doc.sents]
    return sentences

def extract_relationships(sentence_doc,matcher, causal_verbs):
    matches = matcher(sentence_doc)
    # extract (cause, verb, effect) triplets from a sentence, filtering for causal verbs
    rels = [(sentence_doc[match[1][1]].text,sentence_doc[match[1][0]].text,sentence_doc[match[1][2]].text)for match in matches if sentence_doc[match[1][0]].lemma_ in causal_verbs] 
    return rels

def entox_parse(json_string, nlp=nlp,matcher=matcher, causal_verbs=causal_verbs): 
    # take a json string as input and convert it to "regular" text first
    parsed_json = json.loads(json_string)
    #title = parsed_json["MedlineCitation"]["Article"].get("ArticleTitle", "")
    abstract = parsed_json["MedlineCitation"]["Article"].get("AbstractText", "")
    # Then apply NLP
    doc = nlp(abstract)
    sentences = extract_sentences(doc)
    docs = [nlp(sentence) for sentence in sentences]
    relations = [extract_relationships(sentence_doc,matcher, causal_verbs) for sentence_doc in docs]
    # remove empty lists and flatten
    relations = [item for sublist in relations for item in sublist]
    return relations



entox_parse_udf = F.udf(entox_parse, ArrayType(StringType()))

sample_df = sample_df.withColumn("relationships", entox_parse_udf(F.col("abstract")))
# How do we want to extract relations? 
# Duplicate rows for each relationship, and 3 columns per relationship? 
# Put the text of the sentence only as extra column?
# Does it then make sense to make differnt udf functions for all sub-functions?
# make sure if abstract returns nothing it is skipped

sample_df.show(5)

### ERROR MESSAGE ###
"""at org.apache.spark.api.python.BasePythonRunner$WriterThread.run(PythonRunner.scala:282)
Caused by: java.lang.RuntimeException: Cannot reserve additional contiguous bytes in the
vectorized reader (requested 50686118 bytes). As a workaround, you can reduce the vectorized 
reader batch size, or disable the vectorized reader, or disable spark.sql.sources.bucketing.enabled 
if you read from bucket table. For Parquet file format, refer to spark.sql.parquet.columnarReaderBatchSize 
(default 4096) and spark.sql.parquet.enableVectorizedReader; for ORC file format, refer to 
spark.sql.orc.columnarReaderBatchSize (default 4096) and spark.sql.orc.enableVectorizedReader."""