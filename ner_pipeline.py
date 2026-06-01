import pandas as pd
import spacy
from transformers import pipeline as hf_pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc
import os

TARGET_LABELS = {"ORG", "LOC", "EVENT"}

LABEL_MAP = {
    # spaCy
    "ORG":    "ORG",
    "GPE":    "LOC",
    "LOC":    "LOC",
    "EVENT":  "EVENT",
    "FAC":    "LOC",
    # Transformers dslim/bert-base-NER
    "B-ORG":  "ORG", "I-ORG": "ORG",
    "B-LOC":  "LOC", "I-LOC": "LOC",
    "B-MISC": None,  "I-MISC": None,
    "B-PER":  None,  "I-PER":  None,
}


# ─────────────────────────────────────────
# 1. NER spaCy
# ─────────────────────────────────────────
def ner_spacy(df):
    print("\nload model spaCy en_core_web_trf...")
    nlp     = spacy.load("en_core_web_trf")
    results = []

    print(f"mulai ekstraksi spaCy dari {len(df)} data...")
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(f"  spaCy progress: {i}/{len(df)}")

        doc  = nlp(str(row["content_clean"]))
        seen = set()

        for ent in doc.ents:
            label = LABEL_MAP.get(ent.label_)
            word  = ent.text.strip()
            key   = (word, label)

            if label in TARGET_LABELS and key not in seen and len(word) > 2:
                seen.add(key)
                results.append({
                    "post_id":   row["id"],
                    "entity":    word,
                    "label":     label,
                    "model":     "spaCy",
                    "subreddit": row["subreddit"],
                    "date":      row["created"]
                })

    df_spacy = pd.DataFrame(results)
    print(f"\nspaCy selesai, ketemu {len(df_spacy)} entitas")
    print(df_spacy["label"].value_counts())
    return df_spacy


# ─────────────────────────────────────────
# 2. NER Transformers
# ─────────────────────────────────────────
def ner_transformers(df):
    print("\nload model dslim/bert-base-NER...")
    ner = hf_pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="first"
    )

    results = []

    print(f"mulai ekstraksi Transformers dari {len(df)} data...")
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(f"  Transformers progress: {i}/{len(df)}")

        text = str(row["content_clean"])[:512]
        try:
            entities = ner(text)
        except Exception:
            continue

        seen = set()
        for ent in entities:
            label = LABEL_MAP.get(ent["entity_group"])
            word  = ent["word"].strip()
            key   = (word, label)

            if label in TARGET_LABELS and key not in seen and len(word) > 2:
                seen.add(key)
                results.append({
                    "post_id":   row["id"],
                    "entity":    word,
                    "label":     label,
                    "model":     "Transformers",
                    "subreddit": row["subreddit"],
                    "date":      row["created"]
                })

    df_trans = pd.DataFrame(results)
    print(f"\nTransformers selesai, ketemu {len(df_trans)} entitas")
    print(df_trans["label"].value_counts())
    return df_trans


# ─────────────────────────────────────────
# 3. AGREGASI DENGAN SPARK
# ─────────────────────────────────────────
def aggregate_spark(df_all):
    print("\njalankan agregasi pakai Spark...")

    spark = SparkSession.builder \
        .appName("NER_Aggregation") \
        .config("spark.driver.memory", "4g") \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    df = spark.createDataFrame(df_all)

    print("\ntop 10 ORG:")
    df.filter(col("label") == "ORG") \
      .groupBy("entity", "model") \
      .agg(count("*").alias("freq")) \
      .orderBy(desc("freq")) \
      .show(10)

    print("\ntop 10 LOC:")
    df.filter(col("label") == "LOC") \
      .groupBy("entity", "model") \
      .agg(count("*").alias("freq")) \
      .orderBy(desc("freq")) \
      .show(10)

    print("\ntop 10 EVENT:")
    df.filter(col("label") == "EVENT") \
      .groupBy("entity", "model") \
      .agg(count("*").alias("freq")) \
      .orderBy(desc("freq")) \
      .show(10)

    print("\nperbandingan model:")
    df.groupBy("model", "label") \
      .agg(count("*").alias("total")) \
      .orderBy("model", "label") \
      .show()

    # simpan agregasi
    df_agg = df.groupBy("entity", "label", "model") \
               .agg(count("*").alias("freq")) \
               .orderBy(desc("freq"))

    df_agg.toPandas().to_csv("ner_aggregated.csv", index=False)

    spark.stop()
    print("\nagregasi selesai, disimpan ke ner_aggregated.csv")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("baca reddit_clean.csv...")
    df = pd.read_csv("reddit_clean.csv")
    print(f"total data: {len(df)}")

    print("\n" + "=" * 50)
    print("STEP 1: NER spaCy")
    print("=" * 50)
    df_spacy = ner_spacy(df)

    print("\n" + "=" * 50)
    print("STEP 2: NER Transformers")
    print("=" * 50)
    df_trans = ner_transformers(df)

    print("\n" + "=" * 50)
    print("STEP 3: gabung hasil")
    print("=" * 50)
    df_all = pd.concat([df_spacy, df_trans], ignore_index=True)
    df_all.to_csv("ner_results.csv", index=False)
    print(f"total entitas: {len(df_all)}")
    print(df_all["label"].value_counts())

    print("\n" + "=" * 50)
    print("STEP 4: agregasi Spark")
    print("=" * 50)
    aggregate_spark(df_all)

    print("\nselesai semua! file output:")
    print("  ner_results.csv     -> semua entitas kedua model")
    print("  ner_aggregated.csv  -> frekuensi tiap entitas")