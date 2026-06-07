# preprocessing.py
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, trim, length, count, desc

spark = SparkSession.builder \
    .appName("NER_Preprocessing") \
    .config("spark.driver.memory", "4g") \
    .config("spark.hadoop.fs.defaultFS", "file:///") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# baca data mentah
path = os.path.abspath("data/reddit_raw.csv")
print(f"baca data dari {path}...")

df = spark.read.csv(
    path,
    header=True,
    inferSchema=True,
    multiLine=True,
    escape='"'
)

total_awal = df.count()
print(f"total data mentah: {total_awal}")

# hapus duplikat berdasarkan id
print("\nhapus duplikat id...")
before = df.count()
df = df.dropDuplicates(["id"])
print(f"duplikat id dihapus: {before - df.count()} baris")

# hapus duplikat berdasarkan konten
print("\nhapus duplikat konten...")
before = df.count()
df = df.dropDuplicates(["content_clean"])
print(f"duplikat konten dihapus: {before - df.count()} baris")

# filter konten rusak
print("\nfilter konten rusak...")
before = df.count()
df = df.filter(col("content_clean").isNotNull())
df = df.filter(col("subreddit").isNotNull())
df = df.filter(col("content_clean") != "[deleted]")
df = df.filter(col("content_clean") != "[removed]")
df = df.filter(length(col("content_clean")) > 10)
print(f"konten rusak dihapus: {before - df.count()} baris")

# bersihkan sisa noise
print("\nbersihkan sisa noise...")
df = df.withColumn("content_clean",
        regexp_replace(col("content_clean"), r"http\S+", ""))
df = df.withColumn("content_clean",
        regexp_replace(col("content_clean"), r"&[a-z]+;", " "))
df = df.withColumn("content_clean",
        regexp_replace(col("content_clean"), r"\s+", " "))
df = df.withColumn("content_clean",
        trim(col("content_clean")))

# filter lagi setelah cleaning
df = df.filter(length(col("content_clean")) > 10)

# filter subreddit yang tidak valid
# pastikan subreddit hanya huruf, angka, underscore
df = df.filter(col("subreddit").rlike("^[a-zA-Z0-9_]+$"))

final_count = df.count()
print(f"\nselesai preprocessing")
print(f"data awal  : {total_awal}")
print(f"data bersih: {final_count}")
print(f"terhapus   : {total_awal - final_count}")

# info distribusi subreddit
print("\ndistribusi subreddit setelah preprocessing:")
df.groupBy("subreddit") \
  .agg(count("*").alias("jumlah")) \
  .orderBy(desc("jumlah")) \
  .show(50)

# simpan ke csv
output_path = os.path.abspath("data/reddit_clean.csv")
df_pandas = df.toPandas()
df_pandas.to_csv(output_path, index=False, quoting=1)
print(f"\ndata disimpan ke data/reddit_clean.csv")
print(f"total baris tersimpan: {len(df_pandas)}")

spark.stop()