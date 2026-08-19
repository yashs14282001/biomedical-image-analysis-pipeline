# Data directory

The source dataset is not duplicated in this repository. Run:

```bash
python scripts/00_download_dataset.py
```

This downloads `nuclei_dataset.zip` from the assignment repository into `data/raw/` and extracts it.
Then run:

```bash
python scripts/01_prepare_eda.py
```

The preparation step automatically discovers common paired image/mask layouts, converts images to grayscale, resizes images and masks to 256×256, creates/respects train-validation-test splits, and writes `data/processed/manifest.csv`.
