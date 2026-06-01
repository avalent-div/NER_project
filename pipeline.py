import subprocess
import sys
import os

os.makedirs("data", exist_ok=True)

steps = [
    ("STEP 1 - Scraping Data Reddit",       "scraper.py"),
    ("STEP 2 - Preprocessing Apache Spark", "preprocessing.py"),
    ("STEP 3 - NER Pipeline spaCy + BERT",  "ner_pipeline.py"),
]

for label, script in steps:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n[ERROR] {script} gagal! Pipeline dihentikan.")
        sys.exit(1)
    print(f"\n[OK] {script} selesai.")

print("\n" + "="*60)
print("  SEMUA PIPELINE SELESAI!")
print("  Output tersimpan di folder data/")
print("="*60)
