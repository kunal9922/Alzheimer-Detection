# Alzheimer-Detection

A benchmark comparing 10 scikit-learn classifiers on `Alzheimer_Dataset_Details.csv`, with per-algorithm scripts, GridSearchCV tuning, and matplotlib visualizations for side-by-side comparison.

> **About the dataset**: `Alzheimer_Dataset_Details.csv` is an *image manifest* (one row per MRI slice with its filename, dimensions, and file size) — not a clinical feature table. `Width`, `Height`, `Channels`, `Color Mode`, and `Extension` are constant across every row, so they carry no signal and are dropped automatically. Only `File Size (KB)` and a numeric index parsed from the filename vary at all, and neither correlates meaningfully with the diagnosis label. Every algorithm below is trained on the exact same features and train/test split, so the comparison between algorithms is fair — but expect all accuracies to land close to the ~50% majority-class baseline, since the real diagnostic signal lives in the pixel data, not this metadata.

## Project structure

```
ml_algorithms/
├── data_preprocessing.py        # shared loader: feature engineering + train/test split (used by every script below)
├── 01_logistic_regression.py
├── 02_knn.py
├── 03_svm.py
├── 04_decision_tree.py
├── 05_random_forest.py
├── 06_gradient_boosting.py
├── 07_adaboost.py
├── 08_naive_bayes.py
├── 09_lda.py
├── 10_mlp_neural_network.py
├── benchmark_comparison.py      # aggregates all results + generates charts
├── benchmark_summary.csv        # output: summary table (generated)
├── results/                     # output: one JSON of metrics per algorithm (generated)
└── plots/                       # output: PNG charts (generated)
```

## Requirements

- Python 3.10+
- pip packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`

Install them with:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Setup

1. Get `Alzheimer_Dataset_Details.csv` (the image manifest the scripts expect — columns: `Split, Disease, Filename, Extension, Width, Height, Channels, Color Mode, File Size (KB), File Size (MB), Full Path`).
2. Point the scripts at it: open [ml_algorithms/data_preprocessing.py](ml_algorithms/data_preprocessing.py) and set `DATA_PATH` (near the top of the file) to the CSV's location on your machine, e.g.:

   ```python
   DATA_PATH = Path(r"C:\path\to\Alzheimer_Dataset_Details.csv")
   ```

## How to run

All commands are run from inside the `ml_algorithms/` folder.

**Run a single algorithm** — prints a classification report to the console and writes its metrics to `results/<algorithm>.json`:

```bash
cd ml_algorithms
python 05_random_forest.py
```

Each of the 10 numbered scripts (`01_logistic_regression.py` … `10_mlp_neural_network.py`) can be run the same way, independently, in any order.

**Run everything and build the comparison charts** — run all 10 algorithm scripts first, then generate the benchmark:

```bash
cd ml_algorithms
for f in 0*.py 1*.py; do python "$f"; done   # bash; on Windows PowerShell: foreach ($f in Get-ChildItem *.py | Where-Object {$_.Name -match '^\d'}) { python $f.Name }
python benchmark_comparison.py
```

`benchmark_comparison.py` reads every `results/*.json` and produces:
- `benchmark_summary.csv` — accuracy/precision/recall/F1/training time for all 10 algorithms, sorted by accuracy
- `plots/01_accuracy_comparison.png` — accuracy bar chart with the majority-class baseline marked
- `plots/02_metric_comparison.png` — accuracy/precision/recall/F1 grouped by algorithm
- `plots/03_training_time.png` — training time comparison (log scale)
- `plots/04_best_model_confusion_matrix.png` — confusion matrix for the top-performing model

## License

See [LICENSE](LICENSE).
