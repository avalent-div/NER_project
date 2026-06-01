# Named Entity Recognition pada Media Sosial Reddit
> Menggunakan spaCy dan BERT dengan Apache Spark

## Deskripsi
Project Big Data untuk mengekstraksi entitas bernama (ORG, LOC, EVENT)
dari 22.500 postingan Reddit menggunakan dua model NER: spaCy dan BERT,
dengan preprocessing menggunakan Apache Spark.

## Anggota Kelompok
| Nama | NIM |
|------|-----|
| Avalent Divasea | 243016019 |
| Yosephina Ota | 243016004 |

**Mata Kuliah:** Big Data
**Dosen:** Endang Anggiratih, ST, M.Cs
**Universitas Pignateli Triputra 2026**

## Arsitektur Sistem
Reddit JSON API → scraper.py → preprocessing.py (Spark) → ner_pipeline.py (spaCy+BERT) → Flask Dashboard

## Struktur Folder
    NER_project/
    ├── scraper.py
    ├── preprocessing.py
    ├── ner_pipeline.py
    ├── pipeline.py
    ├── requirements.txt
    ├── README.md
    ├── data/
    └── app/

## Cara Menjalankan

Install dependencies:

    pip install -r requirements.txt
    python -m spacy download en_core_web_trf
    python -m spacy download xx_ent_wiki_sm

Jalankan pipeline lengkap:

    python pipeline.py

Atau per tahap:

    python scraper.py
    python preprocessing.py
    python ner_pipeline.py

## Dataset
- **Sumber:** Reddit JSON API (publik)
- **Total:** 22.500 postingan
- **Subreddit:** 24 subreddit (worldnews, indonesia, technology, dll)
- **Rentang:** Desember 2012 - Mei 2026
- **Entitas:** ORG, LOC, EVENT

## Model NER
| Model | Library | Keterangan |
|-------|---------|------------|
| en_core_web_trf | spaCy | Transformer-based, support EVENT |
| dslim/bert-base-NER | HuggingFace | BERT fine-tuned untuk NER |
