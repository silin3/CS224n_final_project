import os, sys, csv
cwd = os.getcwd()
sys.path = [p for p in sys.path if p not in ('', cwd)]
from datasets import load_dataset

ds = load_dataset('paws', 'unlabeled_final')
os.makedirs('data', exist_ok=True)

for split, out in [
    ('train', 'data/paws-unlabeled-final-train.csv'),
    ('validation', 'data/paws-unlabeled-final-validation.csv'),
]:
    rows = ds[split]
    with open(out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['id', 'sentence1', 'sentence2', 'is_duplicate'])
        for r in rows:
            w.writerow([str(r['id']).strip().lower(), r['sentence1'], r['sentence2'], int(r['label'])])
    print(out, len(rows))