import biobricks as bb
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructType, StructField, IntegerType, ArrayType
import spacy
from spacy.matcher import DependencyMatcher
from spacy.language import Language
from negspacy.negation import Negex
from typing import List

# Define spark session - needs enough memory to go through entire PubMed

spark = SparkSession.builder\
    .appName("entox run on pubmed")\
    .config("spark.executor.memory", "32g")\
    .config("spark.driver.memory", "32g")\
    .config("spark.executor.memoryOverhead", "4g")\
    .config("spark.driver.memoryOverhead", "2g")\
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
        #de-nest title
        if isinstance(title,dict):
            title = title["#text"]
        abstract = parsed_json["MedlineCitation"]["Article"].get("Abstract",{}).get("AbstractText", "")
        #'\n'.join(abstract) if isinstance(abstract, list) else \
        #   '\n'.join(abstract.values()) if isinstance(abstract, dict) else abstract
        if isinstance(abstract, list) and '@Label' in abstract[0]:
            # De-nest abstract text further 
            abstract = " ".join([item["#text"] for item in abstract])
        elif isinstance(abstract,dict) and '#text' in abstract:
            # De-nest for other format exception
            abstract = abstract["#text"]
        # Make sure to return a string, even if the title/abstract is missing
        return f"{title}", f"{abstract}"
    except Exception as e:
        # Log the error for debugging
        print("Error:", e)
        return "Error", str(e)


# Register UDF with Spark
build_abstract_udf = F.udf(build_abstracttext, returnType=StructType([
        StructField("title", StringType()),
        StructField("abstract", StringType())
    ]))

# Apply the UDF to create new columns
df_abstract = df.withColumn("parsed_columns", build_abstract_udf(F.col("json")))
df_abstract = df_abstract.select(
    F.col("PMID"),
    F.col("parsed_columns.title").alias("title"),
    F.col("parsed_columns.abstract").alias("abstract")
    )
# Concatenate title and abstract with a newline in between
df_abstract = df_abstract.withColumn('text', F.concat_ws('\n', df_abstract.title, df_abstract.abstract)) 


# 3. RUN NLP ENTOX MODEL ON ALL ABSTRACTS TEXT ==============================================

# Running below once to package the model in one go with merge_entities
# load spacy model
# nlp = spacy.load("en_tox")
# Add merge entities so that the semantic parser for relex functions properly
# nlp.add_pipe("merge_entities", after='ner')
# Add negation to find out if phenotypes are *not* happening?
# doesn't seem to be working properly
# nlp.add_pipe("negex", config={"ent_types":["PHENOTYPE"]}, after="merge_entities")
# nlp.to_disk("/path/to/en_tox_merge")

# Load spaCy model with merge entities module
nlp = spacy.load("en_tox_merge")

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

# Define entities for relationships of interest and causal verbs
matcher = dependency_matcher(nlp, "COMPOUND", "PHENOTYPE")
causal_verbs = ['increase', 'produce', 'cause', 'induce', 'generate', 'effect', 'provoke', 'arouse', 'elicit', 'lead', 'trigger','derive', 'associate', 'relate', 'link', 'stem', 'originate', 'lead', 'bring', 'result', 'inhibit', 'elevate', 'diminish', "exacerbate", "decrease"]


# Relationship extraction functions
def extract_sentences(doc) -> List[str]:  
    sentences = [sent.text for sent in doc.sents]
    return sentences

def extract_relationships(sentence_doc,matcher, causal_verbs):
    matches = matcher(sentence_doc)
    # extract (cause, verb, effect) triplets from a sentence, filtering for causal verbs
    rels = [(sentence_doc[match[1][1]].text,sentence_doc[match[1][0]].lemma_,sentence_doc[match[1][2]].text)for match in matches if sentence_doc[match[1][0]].lemma_ in causal_verbs] 
    return rels

def entox_parse(text, nlp=nlp,matcher=matcher, causal_verbs=causal_verbs): 
    # Then apply NLP
    doc = nlp(text)
    sentences = extract_sentences(doc)
    docs = [nlp(sentence) for sentence in sentences]
    relations = [extract_relationships(sentence_doc,matcher, causal_verbs) for sentence_doc in docs]
    # remove empty lists and flatten
    relations = [item for sublist in relations for item in sublist]
    return relations

# Define a schema that matches the relationships tuple's structure
schema_rels = StructType([
    StructField("cause", StringType(), nullable=False),
    StructField("verb", StringType(), nullable=False),
    StructField("effect", StringType(), nullable=False)
])

# Create UDF
entox_parse_udf = F.udf(entox_parse, ArrayType(schema_rels))

df_rel = sample_df.withColumn("relationships", entox_parse_udf(F.col("text")))

# Filter for abstracts that do contain relationships
df_rel_filtered = df_rel.filter(F.size(df_rel["relationships"])>0)

# Save abstracts + relationships in parquet file
df_rel_filtered.write.parquet("rels.parquet")


df_rel_filtered = spark.read.parquet("rels.parquet")

# 1 relationship per row
df_exploded = df_rel_filtered.withColumn("relationship", F.explode(F.col("relationships")))

# Select the exploded "relationship" column and expand its fields
df_final = df_exploded.select(
    F.col("PMID"),
    F.col("title"),
    F.col("abstract"),
    F.col("relationship.cause").alias("cause"),
    F.col("relationship.verb").alias("verb"),
    F.col("relationship.effect").alias("effect")
)

# Harmonize entities so that they are all capitalized
def capitalize_words(text):
    # Split the text into words
    words = text.split()
    # Capitalize the first letter of each word if it's not all uppercase
    capitalized_words = [word.capitalize() if not word.isupper() else word for word in words]
    # Join the words back into a single string
    return ' '.join(capitalized_words)

# Register the UDF with PySpark
capitalize_words_udf = F.udf(capitalize_words, StringType())

# Apply to dataframe
df_final = df_final.withColumn("cause", capitalize_words_udf(F.col("cause"))) \
                   .withColumn("effect", capitalize_words_udf(F.col("verb")))

# Save dataframe
# df_final.write.parquet("rels.parquet")

#TODO:
# Load into a neo4j instance to inspect results? Maybe at least liver cases
# Or is there a better way to investigate results?

# Stop spark session
spark.stop()
