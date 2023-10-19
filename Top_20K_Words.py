from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, lower
from pyspark.sql.types import StringType
from pyspark.sql import functions as F
import numpy as np
import pandas as pd
import sys

def is_float(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: top_taxis.py <input_file> <output_file>")
    sys.exit(1)

  input_file1 = sys.argv[1]
  input_file2 = sys.argv[2]

  spark = SparkSession.builder.appName("Assignment4_Ques1").getOrCreate()

  wikipedia_pages = spark.read.text(input_file1)

  words_df = wikipedia_pages.select(explode(split(lower(wikipedia_pages.value), "\s+")).alias("word"))

  words_df = words_df.withColumn("word", F.regexp_replace("word", "[^a-zA-Z]", ""))
  words_df = words_df.filter(words_df.word != "")
  word_counts = words_df.groupBy("word").count()
  sorted_word_counts = word_counts.orderBy(F.desc("count"))
  top_words = sorted_word_counts.limit(20000)
  top_words_array = top_words.select("word").rdd.flatMap(lambda x: x).collect()
  print(top_words_array)

  wiki_df = spark.read.option("header", "false").csv(input_file2)
  wiki_df.head()
  wiki_df = wiki_df.withColumnRenamed("_c0", "docID").withColumnRenamed("_c1", "text")
  wiki_df.head()
  def process_document(row):
    docID = row["docID"]
    text = row["text"]
    words = text.split(" ")
    word_positions = np.array([top_words_array.index(word) if word in top_words_array else -1 for word in words])
    return (docID, word_positions)

  processed_rdd = wiki_df.rdd.map(process_document)
  result = processed_rdd.collect()

  for docID, word_positions in result:
    print(f"DocID: {docID}, Word Positions: {word_positions}")

  spark.stop()