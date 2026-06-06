import os
import json
import pandas as pd
import spacy
from spacy import displacy
from transformers import pipeline as hf_pipeline
from flask import Flask, render_template, request, jsonify, send_file, make_response
from collections import Counter
import io

app = Flask(__name__)
app.jinja_env.globals.update(enumerate=enumerate)

# ─────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

print("Membaca data...")
df_results = pd.read_csv(os.path.join(DATA_DIR, 'ner_results.csv'))
df_agg     = pd.read_csv(os.path.join(DATA_DIR, 'ner_aggregated.csv'))
print(f"Data dimuat: {len(df_results)} entitas dari {df_results['post_id'].nunique()} post")

# ─────────────────────────────────────────
# Load NER Models
# ─────────────────────────────────────────
print("Memuat model spaCy...")
nlp = spacy.load('en_core_web_trf')

print("Memuat model BERT...")
ner_bert = hf_pipeline(
    'ner',
    model='dslim/bert-base-NER',
    aggregation_strategy='first'
)
print("Semua model siap!")

LABEL_MAP = {
    'GPE'  : 'LOC',
    'LOC'  : 'LOC',
    'ORG'  : 'ORG',
    'EVENT': 'EVENT',
    'MISC' : 'EVENT'   # BERT tidak punya label EVENT, MISC dimapping ke EVENT
}
TARGET_LABELS = {'ORG', 'LOC', 'EVENT'}


# ─────────────────────────────────────────
# Helper
# ─────────────────────────────────────────
def get_global_stats():
    by_label = df_results['label'].value_counts().to_dict()
    by_model = df_results['model'].value_counts().to_dict()
    return {
        'total_posts'   : int(df_results['post_id'].nunique()),
        'total_entities': int(len(df_results)),
        'org'           : int(by_label.get('ORG', 0)),
        'loc'           : int(by_label.get('LOC', 0)),
        'event'         : int(by_label.get('EVENT', 0)),
        'spacy'         : int(by_model.get('spaCy', 0)),
        'bert'          : int(by_model.get('Transformers', 0)),
    }


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.route('/')
def index():
    stats = get_global_stats()
    top_subreddits = df_results['subreddit'].value_counts().head(8).to_dict()
    top_entities   = df_agg.nlargest(5, 'freq')[['entity','label','freq']].to_dict('records')
    return render_template('index.html',
        stats=stats,
        top_subreddits=json.dumps(top_subreddits),
        top_entities=top_entities
    )


@app.route('/dashboard')
def dashboard():
    label_filter = request.args.get('label', 'all')
    model_filter = request.args.get('model', 'all')

    df = df_agg.copy()
    if label_filter != 'all':
        df = df[df['label'] == label_filter]
    if model_filter != 'all':
        df = df[df['model'] == model_filter]

    def top10(label):
        d = df[df['label'] == label].nlargest(10, 'freq')
        return d[['entity','freq']].to_dict('records')

    return render_template('dashboard.html',
        top_org=json.dumps(top10('ORG')),
        top_loc=json.dumps(top10('LOC')),
        top_event=json.dumps(top10('EVENT')),
        label_filter=label_filter,
        model_filter=model_filter,
        stats=get_global_stats()
    )


@app.route('/annotator', methods=['GET', 'POST'])
def annotator():
    result     = None
    input_text = ''
    model_used = 'both'

    if request.method == 'POST':
        input_text = request.form.get('text', '').strip()
        model_used = request.form.get('model', 'both')

        if input_text:
            spacy_html    = None
            bert_entities = []

            if model_used in ['spacy', 'both']:
                doc = nlp(input_text)
                colors = {
                    'ORG'   : '#e0e7ff',
                    'LOC'   : '#e0f2fe',
                    'EVENT' : '#d1fae5',
                    'GPE'   : '#e0f2fe',
                    'PERSON': '#fef9c3'
                }
                options = {'colors': colors}
                spacy_html = displacy.render(doc, style='ent', page=False, options=options)

            if model_used in ['bert', 'both']:
                ents = ner_bert(input_text[:512])
                bert_entities = []
                for e in ents:
                    mapped = LABEL_MAP.get(e['entity_group'], e['entity_group'])
                    bert_entities.append({
                        'entity': mapped,
                        'word'  : e['word'],
                        'score' : round(float(e['score']), 3),
                        'original_label': e['entity_group']
                    })

            result = {'spacy_html': spacy_html, 'bert_entities': bert_entities}

    return render_template('annotator.html',
        result=result, input_text=input_text, model_used=model_used,
        stats=get_global_stats()
    )


@app.route('/comparison')
def comparison():
    spacy_df = df_results[df_results['model'] == 'spaCy']
    bert_df  = df_results[df_results['model'] == 'Transformers']

    def counts(df):
        return {l: int(df[df['label'] == l].shape[0]) for l in ['ORG', 'LOC', 'EVENT']}

    spacy_counts = counts(spacy_df)
    bert_counts  = counts(bert_df)

    spacy_ents = set(spacy_df['entity'].str.lower())
    bert_ents  = set(bert_df['entity'].str.lower())

    comparison_data = {}
    for label in ['ORG', 'LOC', 'EVENT']:
        s = spacy_df[spacy_df['label'] == label]['entity'].value_counts().head(5)
        b = bert_df[bert_df['label'] == label]['entity'].value_counts().head(5)
        comparison_data[label] = {
            'spacy': [{'entity': k, 'freq': int(v)} for k, v in s.items()],
            'bert' : [{'entity': k, 'freq': int(v)} for k, v in b.items()],
        }

    return render_template('comparison.html',
        spacy_counts=spacy_counts,
        bert_counts=bert_counts,
        only_spacy=len(spacy_ents - bert_ents),
        only_bert=len(bert_ents - spacy_ents),
        both=len(spacy_ents & bert_ents),
        comparison_data=json.dumps(comparison_data),
        stats=get_global_stats()
    )


@app.route('/network')
def network():
    post_entities = df_results.groupby('post_id')['entity'].apply(list)
    cooccur = Counter()
    for ents in post_entities:
        unique = list(set(ents))[:8]
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                pair = tuple(sorted([unique[i], unique[j]]))
                cooccur[pair] += 1

    top_pairs = cooccur.most_common(40)
    nodes_set = {}
    edges = []
    entity_label = (
        df_results[['entity', 'label']]
        .drop_duplicates()
        .set_index('entity')['label']
        .to_dict()
    )
    color_map = {'ORG': '#6366f1', 'LOC': '#0ea5e9', 'EVENT': '#10b981'}

    for (e1, e2), count in top_pairs:
        for e in [e1, e2]:
            if e not in nodes_set:
                lbl = entity_label.get(e, 'ORG')
                nodes_set[e] = {
                    'id'   : e,
                    'label': e[:25],
                    'color': color_map.get(lbl, '#6366f1'),
                    'title': f"{e} ({lbl})"
                }
        edges.append({
            'from' : e1,
            'to'   : e2,
            'value': count,
            'title': f'Co-occurrence: {count}x'
        })

    return render_template('network.html',
        nodes=json.dumps(list(nodes_set.values())),
        edges=json.dumps(edges),
        stats=get_global_stats()
    )


@app.route('/timeline')
def timeline():
    df = df_results.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['month'] = df['date'].dt.to_period('M').astype(str)

    pivot  = df.groupby(['month', 'label']).size().unstack(fill_value=0).reset_index()
    months = pivot['month'].tolist()

    def safe_list(col):
        return [int(x) for x in pivot.get(col, [0] * len(months)).tolist()]

    trending = df['entity'].value_counts().head(10).reset_index()
    trending.columns = ['entity', 'freq']

    return render_template('timeline.html',
        months=json.dumps(months),
        org_counts=json.dumps(safe_list('ORG')),
        loc_counts=json.dumps(safe_list('LOC')),
        event_counts=json.dumps(safe_list('EVENT')),
        trending=trending.to_dict('records'),
        stats=get_global_stats()
    )


@app.route('/batch', methods=['GET', 'POST'])
def batch():
    result = None

    if request.method == 'POST':
        file         = request.files.get('file')
        model_choice = request.form.get('model', 'spacy')

        if file and file.filename.endswith('.csv'):
            df  = pd.read_csv(file)
            col = 'content_clean' if 'content_clean' in df.columns else df.columns[0]
            output = []

            for _, row in df.head(50).iterrows():
                text = str(row[col])[:512]
                ents = []

                if model_choice in ['spacy', 'both']:
                    doc = nlp(text)
                    for e in doc.ents:
                        mapped = LABEL_MAP.get(e.label_)
                        if mapped in TARGET_LABELS:
                            ents.append({'entity': e.text, 'label': mapped, 'model': 'spaCy'})

                if model_choice in ['bert', 'both']:
                    for e in ner_bert(text):
                        lbl = LABEL_MAP.get(e['entity_group'])
                        if lbl in TARGET_LABELS:
                            ents.append({'entity': e['word'], 'label': lbl, 'model': 'BERT'})

                output.append({'text': text[:80] + '...', 'entities': ents})

            result = {
                'data' : output,
                'total': len(output),
                'json' : json.dumps(output, indent=2, ensure_ascii=False)
            }

    return render_template('batch.html', result=result, stats=get_global_stats())


@app.route('/api/download-json', methods=['POST'])
def download_json():
    data = request.json.get('data', '')
    buf  = io.BytesIO(data.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name='ner_results.json',
                     mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
