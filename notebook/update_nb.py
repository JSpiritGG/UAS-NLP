import json
import scipy
import sys

def read_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_notebook(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  Saved: {path}")

def set_cell_source(nb, cell_index, new_source):
    lines = new_source.split('\n')
    source_lines = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            source_lines.append(line + '\n')
        else:
            source_lines.append(line)
    if cell_index < len(nb['cells']):
        nb['cells'][cell_index]['source'] = source_lines
        if nb['cells'][cell_index].get('cell_type') == 'code':
            nb['cells'][cell_index]['outputs'] = []
            nb['cells'][cell_index]['execution_count'] = None

ml_path = r"d:\Atma Jaya\SEMESTER 6\Pemrosesan Bahasa Alami\uas lengkap\uas lengkap\notebook\Kelp2_multilabel_modeling.ipynb"
ml_nb = read_notebook(ml_path)

# Apply fixes to the notebook
# --- Cell 1 ---
set_cell_source(ml_nb, 1, r'''from pathlib import Path
from collections import Counter
import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    f1_score, hamming_loss, accuracy_score,
    classification_report, confusion_matrix,
)

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _stemmer = StemmerFactory().createStemmer()
    HAS_SASTRAWI = True
    print("[INFO] Sastrawi stemmer tersedia.")
except ImportError:
    HAS_SASTRAWI = False
    _stemmer = None
    print("[WARN] Sastrawi tidak tersedia, stemming dilewati.")

try:
    import nltk
    try:
        from nltk.corpus import stopwords as nltk_stopwords
        STOP_WORDS_ID = set(nltk_stopwords.words('indonesian'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords as nltk_stopwords
        STOP_WORDS_ID = set(nltk_stopwords.words('indonesian'))
    HAS_NLTK_STOPWORDS = True
    print(f"[INFO] NLTK stopwords tersedia ({len(STOP_WORDS_ID)} kata).")
except (ImportError, LookupError):
    STOP_WORDS_ID = set()
    HAS_NLTK_STOPWORDS = False
    print("[WARN] NLTK stopwords tidak tersedia.")

SEED = 42
np.random.seed(SEED)''')

# --- Cell 11 ---
set_cell_source(ml_nb, 11, r'''def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^0-9a-zA-Z_\-\s]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def advanced_clean_text(text: str) -> str:
    text = clean_text(text)
    if HAS_NLTK_STOPWORDS and STOP_WORDS_ID:
        words = text.split()
        words = [w for w in words if w not in STOP_WORDS_ID]
        text = ' '.join(words)
    if HAS_SASTRAWI and _stemmer is not None:
        text = _stemmer.stem(text)
    return text

for df in [train_df, val_df, test_df]:
    df['text_clean'] = df['text'].apply(clean_text)
    df['text_advanced'] = df['text'].apply(advanced_clean_text)
    df['char_length'] = df['text_clean'].str.len()
    df['word_count'] = df['text_clean'].str.split().apply(len)

comparison = train_df[['text', 'text_clean', 'text_advanced']].head(5)
display(comparison)''')

# --- Cell 25 ---
set_cell_source(ml_nb, 25, r'''def make_tfidf_logreg(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000, ngram_range=(1, 3), min_df=2, sublinear_tf=True)),
        ('classifier', OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight='balanced', C=C, solver='liblinear'))),
    ])

def make_tfidf_svm(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000, ngram_range=(1, 3), min_df=2, sublinear_tf=True)),
        ('classifier', OneVsRestClassifier(LinearSVC(class_weight='balanced', C=C, max_iter=5000))),
    ])

def make_word_char_tfidf_logreg(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ('features', FeatureUnion([
            ('word_tfidf', TfidfVectorizer(max_features=8000, ngram_range=(1, 3), min_df=2, sublinear_tf=True, analyzer='word')),
            ('char_tfidf', TfidfVectorizer(max_features=5000, ngram_range=(2, 5), min_df=2, sublinear_tf=True, analyzer='char_wb')),
        ])),
        ('classifier', OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight='balanced', C=C, solver='liblinear'))),
    ])

def make_word_char_tfidf_svm(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ('features', FeatureUnion([
            ('word_tfidf', TfidfVectorizer(max_features=8000, ngram_range=(1, 3), min_df=2, sublinear_tf=True, analyzer='word')),
            ('char_tfidf', TfidfVectorizer(max_features=5000, ngram_range=(2, 5), min_df=2, sublinear_tf=True, analyzer='char_wb')),
        ])),
        ('classifier', OneVsRestClassifier(LinearSVC(class_weight='balanced', C=C, max_iter=5000))),
    ])

def make_tfidf_rf(n_estimators: int = 200) -> Pipeline:
    return Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000, ngram_range=(1, 3), min_df=2, sublinear_tf=True)),
        ('classifier', OneVsRestClassifier(RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', random_state=42, n_jobs=-1))),
    ])

def make_bow_chain_logreg(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ('vectorizer', CountVectorizer(max_features=10000, ngram_range=(1, 3), min_df=2)),
        ('classifier', ClassifierChain(LogisticRegression(max_iter=2000, class_weight='balanced', C=C, solver='liblinear'), order='random', random_state=42)),
    ])''')

# --- Cell 29 ---
set_cell_source(ml_nb, 29, r'''EXPERIMENT_SPECS = []

for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
    EXPERIMENT_SPECS.append({'name': f'TFIDF_1-3gram_OVR_LogReg_C{C}', 'factory': lambda C=C: make_tfidf_logreg(C), 'text_key': 'basic', 'representation': 'TF-IDF word 1-3gram (sublinear)', 'problem_transformation': 'Binary Relevance / OneVsRest', 'algorithm': 'Logistic Regression', 'tuned_param': f'C={C}'})
    EXPERIMENT_SPECS.append({'name': f'TFIDF_1-3gram_OVR_SVM_C{C}', 'factory': lambda C=C: make_tfidf_svm(C), 'text_key': 'basic', 'representation': 'TF-IDF word 1-3gram (sublinear)', 'problem_transformation': 'Binary Relevance / OneVsRest', 'algorithm': 'Linear SVM', 'tuned_param': f'C={C}'})

for C in [0.5, 1.0, 2.0, 5.0]:
    EXPERIMENT_SPECS.append({'name': f'WordChar_TFIDF_OVR_LogReg_C{C}', 'factory': lambda C=C: make_word_char_tfidf_logreg(C), 'text_key': 'basic', 'representation': 'Word+Char TF-IDF FeatureUnion', 'problem_transformation': 'Binary Relevance / OneVsRest', 'algorithm': 'Logistic Regression', 'tuned_param': f'C={C}'})

for C in [0.5, 1.0, 2.0]:
    EXPERIMENT_SPECS.append({'name': f'WordChar_TFIDF_OVR_SVM_C{C}', 'factory': lambda C=C: make_word_char_tfidf_svm(C), 'text_key': 'basic', 'representation': 'Word+Char TF-IDF FeatureUnion', 'problem_transformation': 'Binary Relevance / OneVsRest', 'algorithm': 'Linear SVM', 'tuned_param': f'C={C}'})

for n_est in [100, 200, 300]:
    EXPERIMENT_SPECS.append({'name': f'TFIDF_1-3gram_OVR_RF_n{n_est}', 'factory': lambda n=n_est: make_tfidf_rf(n), 'text_key': 'basic', 'representation': 'TF-IDF word 1-3gram (sublinear)', 'problem_transformation': 'Binary Relevance / OneVsRest', 'algorithm': 'Random Forest', 'tuned_param': f'n_estimators={n_est}'})

for C in [0.5, 1.0, 2.0]:
    EXPERIMENT_SPECS.append({'name': f'BOW_1-3gram_Chain_LogReg_C{C}', 'factory': lambda C=C: make_bow_chain_logreg(C), 'text_key': 'basic', 'representation': 'Bag of Words 1-3gram', 'problem_transformation': 'Classifier Chain', 'algorithm': 'Logistic Regression', 'tuned_param': f'C={C}'})

for C in [0.5, 1.0, 2.0]:
    EXPERIMENT_SPECS.append({'name': f'ADV_TFIDF_LogReg_C{C}', 'factory': lambda C=C: make_tfidf_logreg(C), 'text_key': 'advanced', 'representation': 'Advanced Text + TF-IDF 1-3gram', 'problem_transformation': 'Binary Relevance / OneVsRest', 'algorithm': 'Logistic Regression', 'tuned_param': f'C={C}'})
    EXPERIMENT_SPECS.append({'name': f'ADV_TFIDF_SVM_C{C}', 'factory': lambda C=C: make_tfidf_svm(C), 'text_key': 'advanced', 'representation': 'Advanced Text + TF-IDF 1-3gram', 'problem_transformation': 'Binary Relevance / OneVsRest', 'algorithm': 'Linear SVM', 'tuned_param': f'C={C}'})
    EXPERIMENT_SPECS.append({'name': f'ADV_WordChar_LogReg_C{C}', 'factory': lambda C=C: make_word_char_tfidf_logreg(C), 'text_key': 'advanced', 'representation': 'Advanced Text + Word+Char TF-IDF', 'problem_transformation': 'Binary Relevance / OneVsRest', 'algorithm': 'Logistic Regression', 'tuned_param': f'C={C}'})

print(f"Total experiments: {len(EXPERIMENT_SPECS)}")
pd.DataFrame([{k: v for k, v in spec.items() if k not in ('factory',)} for spec in EXPERIMENT_SPECS])''')

# --- Cell 31 ---
set_cell_source(ml_nb, 31, r'''trained_models = {}
experiment_rows = []

for i, spec in enumerate(EXPERIMENT_SPECS):
    name = spec['name']
    text_key = spec.get('text_key', 'basic')
    print(f"[{i+1}/{len(EXPERIMENT_SPECS)}] Training: {name}...")
    if text_key == 'advanced':
        X_tr = train_df['text_advanced'].tolist()
        X_va = val_df['text_advanced'].tolist()
    else:
        X_tr = X_train
        X_va = X_val
    try:
        model = spec['factory']()
        model.fit(X_tr, y_train)
        metrics = evaluate_multilabel(model, X_va, y_val, name, 'validation')
        metrics.update({'representation': spec['representation'], 'problem_transformation': spec['problem_transformation'], 'algorithm': spec['algorithm'], 'tuned_param': spec['tuned_param']})
        trained_models[name] = (model, text_key)
        experiment_rows.append(metrics)
        print(f"   micro_f1={metrics['micro_f1']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    except Exception as e:
        print(f"   Error: {e}")

exp_df = pd.DataFrame(experiment_rows).sort_values(by=['micro_f1', 'macro_f1', 'subset_accuracy'], ascending=False).reset_index(drop=True)
exp_df.to_csv(RESULTS_DIR / 'multilabel_experiment_results_val.csv', index=False)
display(exp_df)''')

# --- Cell 33 ---
set_cell_source(ml_nb, 33, r'''best_row = exp_df.iloc[0]
best_model_name = best_row['model']
best_model, best_text_key = trained_models[best_model_name]

print('Best validation model:', best_model_name)
print('Text preprocessing:', best_text_key)
display(best_row.to_frame().T)''')

# --- Cell 35 ---
set_cell_source(ml_nb, 35, r'''if best_text_key == 'advanced':
    X_te = test_df['text_advanced'].tolist()
else:
    X_te = X_test

y_test_pred = best_model.predict(X_te)
test_metrics = evaluate_multilabel(best_model, X_te, y_test, best_model_name, 'test')
print(json.dumps(test_metrics, indent=2))

target_met = test_metrics['micro_f1'] > 0.85
print(f"\n Target micro-F1 > 0.85: {'TERCAPAI!' if target_met else 'Belum tercapai'}")
print(f"   Actual micro-F1: {test_metrics['micro_f1']:.4f}")

with open(RESULTS_DIR / 'multilabel_best_test_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(test_metrics, f, indent=2)

report_dict = classification_report(y_test, y_test_pred, target_names=LABELS, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).T
report_df.to_csv(RESULTS_DIR / 'multilabel_classification_report.csv')

print(classification_report(y_test, y_test_pred, target_names=LABELS, zero_division=0))
display(report_df)''')

# --- Cell 43 ---
set_cell_source(ml_nb, 43, r'''model_package = {
    'model': best_model,
    'labels': LABELS,
    'best_model_name': best_model_name,
    'cleaning_note': 'lowercase, remove URL, keep alphanumeric/underscore/hyphen, normalize spaces' + (', text_key=advanced, stemming Sastrawi, stopword removal' if best_text_key == 'advanced' else ''),
}
joblib.dump(model_package, MODELS_DIR / 'Kelp2_best_multilabel_model.joblib')
print('Saved to:', (MODELS_DIR / 'Kelp2_best_multilabel_model.joblib').resolve())''')

# --- Cell 45 ---
set_cell_source(ml_nb, 45, r'''def predict_multilabel_review(text: str, model_package: dict, threshold: float = 0.5) -> pd.DataFrame:
    model = model_package['model']
    labels = model_package['labels']
    text_clean = advanced_clean_text(text) if 'text_key=advanced' in model_package.get('cleaning_note', '') else clean_text(text)
    pred = model.predict([text_clean])[0]
    rows = []
    for label, value in zip(labels, pred):
        rows.append({'label': label, 'predicted': int(value)})
    return pd.DataFrame(rows)

sample_review = 'Bajunya bagus, bahan adem, harganya juga murah, tapi tempat parkirnya agak sempit.'
sample_pred = predict_multilabel_review(sample_review, model_package)
display(sample_pred[sample_pred['predicted'] == 1])''')

# --- Cell 48 ---
set_cell_source(ml_nb, 48, r'''import joblib
import scipy.special
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, accuracy_score, classification_report, confusion_matrix

ml_payload = joblib.load(MODELS_DIR / 'Kelp2_best_multilabel_model.joblib')
best_model = ml_payload['model']
label_columns = ml_payload['labels']
cleaning_note = ml_payload.get('cleaning_note', '')
text_key = 'text_advanced' if 'text_key=advanced' in cleaning_note else 'text_clean'

print(f"Menggunakan kolom teks: {text_key}")
X_val = val_df[text_key].fillna('').astype(str).tolist()
y_val = val_df[label_columns].values
X_test = test_df[text_key].fillna('').astype(str).tolist()
y_test = test_df[label_columns].values

def multilabel_metrics(y_true, y_pred):
    return {
        'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'hamming_loss': hamming_loss(y_true, y_pred),
        'subset_accuracy': accuracy_score(y_true, y_pred),
    }

default_test_pred = best_model.predict(X_test)
default_test_metrics = multilabel_metrics(y_test, default_test_pred)
default_test_metrics''')

# --- Cell 50 ---
set_cell_source(ml_nb, 50, r'''def get_scores(model, X):
    if hasattr(model, 'predict_proba'):
        scores = model.predict_proba(X)
        if isinstance(scores, list):
            scores = np.vstack([score[:, 1] for score in scores]).T
        return scores
    elif hasattr(model, 'decision_function'):
        d_scores = model.decision_function(X)
        return scipy.special.expit(d_scores)
    else:
        return model.predict(X)

def tune_multilabel_thresholds(y_true, scores):
    thresholds = []  
    pred = np.zeros_like(y_true)

    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() == 0:
            best_t = 0.5
        else:
            best_f1, best_t = -1.0, 0.5
            for t in np.arange(0.05, 0.96, 0.05):
                candidate = (scores[:, j] >= t).astype(int)
                score = f1_score(y_true[:, j], candidate, zero_division=0)
                if score > best_f1 or (abs(score - best_f1) < 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
                    best_f1, best_t = score, float(t)
        thresholds.append(best_t)
        pred[:, j] = (scores[:, j] >= best_t).astype(int)

    for i in range(len(pred)):
        if pred[i].sum() == 0:
            pred[i, int(scores[i].argmax())] = 1
    return np.array(thresholds), pred

val_scores = get_scores(best_model, X_val)
thresholds, val_pred_tuned = tune_multilabel_thresholds(y_val, val_scores)
threshold_df = pd.DataFrame({'label': label_columns, 'threshold': thresholds})
display(threshold_df)''')

# --- Cell 51 ---
set_cell_source(ml_nb, 51, r'''test_scores = get_scores(best_model, X_test)

test_pred_tuned = (test_scores >= thresholds).astype(int)
for i in range(len(test_pred_tuned)):
    if test_pred_tuned[i].sum() == 0:
        test_pred_tuned[i, int(test_scores[i].argmax())] = 1

comparison_threshold_df = pd.DataFrame([
    {'setting': 'Default threshold 0.5', **default_test_metrics},
    {'setting': 'Validation tuned threshold', **multilabel_metrics(y_test, test_pred_tuned)},
])
display(comparison_threshold_df)''')

# Clear outputs
for cell in ml_nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

write_notebook(ml_nb, ml_path)
print("Notebook Kelp2_multilabel_modeling.ipynb fully updated!")
