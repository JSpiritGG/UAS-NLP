import json
import re

nb_path = r'd:\Atma Jaya\SEMESTER 6\Pemrosesan Bahasa Alami\uas lengkap\uas lengkap\notebook\Kelp2_multilabel_modeling.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Replace predict_proba for X_val
        if 'val_scores = best_model.predict_proba(X_val)' in source and 'def get_scores' not in source:
            replacement = '''import scipy.special

def get_scores(model, X):
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

val_scores = get_scores(best_model, X_val)'''
            new_source = source.replace('val_scores = best_model.predict_proba(X_val)', replacement)
            
            lines = new_source.split('\n')
            cell['source'] = [line + '\n' if i < len(lines)-1 else line for i, line in enumerate(lines)]
            print("Patched X_val predict_proba.")
            
        # Replace predict_proba for X_test
        if 'test_scores = best_model.predict_proba(X_test)' in source and 'get_scores(best_model, X_test)' not in source:
            new_source = source.replace('test_scores = best_model.predict_proba(X_test)', 'test_scores = get_scores(best_model, X_test)')
            lines = new_source.split('\n')
            cell['source'] = [line + '\n' if i < len(lines)-1 else line for i, line in enumerate(lines)]
            print("Patched X_test predict_proba.")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Done!')
