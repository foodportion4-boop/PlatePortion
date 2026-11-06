"""Check reproducible week gates without pretending pending reviews succeeded."""
from pathlib import Path
import csv
from plateclock.io.loaders import load_subset
p=Path('data/processed/plate_subset.csv')
if not p.exists(): raise SystemExit('Missing 300-image CSV: run selection step')
count=sum(1 for _ in load_subset(p))
if count!=300:raise SystemExit(f'Expected 300 loadable samples, got {count}')
print('Loader: 300 valid images and masks')
for file in ['results/tables/seg_baseline.csv','results/figures/checkpoint1/review.csv']:
    print(file, 'present' if Path(file).exists() else 'PENDING')
