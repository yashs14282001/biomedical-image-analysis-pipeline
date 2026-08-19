# Auditable Hybrid Biomedical Image Analysis of Nuclei

A GitHub-ready implementation of the assignment pipeline:

**raw microscopy image → grayscale/256×256 → direct VLM description → classical Otsu features → small U-Net → region features → structured JSON → constrained narrative → aggregated CSV**

The project is designed for **educational use only**. The language-model outputs are deliberately descriptive rather than diagnostic, and measured/structured values are retained separately from generated narrative text.

## What is included

| Assignment item | Implementation | Main outputs |
|---|---|---|
| Task 1: preparation + EDA + multimodal LLM | `01_prepare_eda.py`, `02_task1_vlm.py` | sample images, intensity histogram, naive VLM output, schema-valid JSON, 3 repeated runs |
| Task 2: classical features + LLM | `03_task2_classical_llm.py` | Otsu mask, cleaned components, `regionprops_table` CSV, numbers-only JSON + narrative |
| Task 3: U-Net | `04_task3_train_unet.py` | checkpoint, loss/Dice/IoU curves, validation panels, mean Dice/IoU, Otsu-vs-U-Net table/examples |
| Task 4: hybrid test pipeline | `05_task4_hybrid_pipeline.py` | predicted masks, per-image feature CSVs, JSON records, narratives, `hybrid_test_records.csv` |
| Extra credit: robustness | `06_extension_robustness.py` | corrupted-image propagation panel + trace CSV + clean/corrupt records/narratives |
| Extra credit: loss ablation | `07_extension_loss_ablation.py` | BCE vs Dice vs BCE+Dice validation table and bar chart |

## 1. VS Code setup

Recommended: Python 3.10–3.12.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you have an NVIDIA GPU and want CUDA acceleration, install the PyTorch build appropriate for your CUDA setup before installing the remaining requirements.

## 2. Install and prepare Ollama

Install Ollama and make sure its local server/app is running. Then pull the two local models used by the project:

```bash
ollama pull llama3.2-vision
ollama pull llama3.2:3b
```

- `llama3.2-vision` is used for the direct image description in Task 1.
- `llama3.2:3b` is used for the numbers-only interpretation and final narrative steps.

The prompt files are stored in `prompts/` so the exact optimised prompts can be copied into the report.

## 3. Download the assignment dataset

```bash
python scripts/00_download_dataset.py
```

Source used by the assignment:

`https://github.com/Nickolay-K/Assingnment-3-dataset`

The loader supports common biomedical layouts, including `images/` + `masks/`, train/validation/test subfolders, and sample folders containing one image plus several object masks.

## 4. Run Tasks 1–4

Run each task separately so you can inspect the outputs after every stage:

```bash
python scripts/01_prepare_eda.py
python scripts/02_task1_vlm.py
python scripts/03_task2_classical_llm.py
python scripts/04_task3_train_unet.py --epochs 20 --loss bce_dice
python scripts/05_task4_hybrid_pipeline.py
```

Or run the core pipeline end-to-end:

```bash
python run_all.py --epochs 20
```

For debugging on a computer where Ollama is not running:

```bash
python run_all.py --epochs 2 --skip-llm
```

`--skip-llm` is only a debugging mode. Run the real LLM steps before submission because the rubric requires the local LLM outputs.

## 5. Extra-credit experiments

### Robustness trace

```bash
python scripts/06_extension_robustness.py --corruption blur
python scripts/06_extension_robustness.py --corruption low_contrast
python scripts/06_extension_robustness.py --corruption noise
```

This saves input-level changes, clean-vs-corrupt U-Net mask Dice, feature changes, and (unless `--no-llm` is used) clean/corrupt structured records and narratives.

### Loss ablation

```bash
python scripts/07_extension_loss_ablation.py --epochs 12
```

This compares **BCE**, **Dice**, and **BCE+Dice** using the same validation split and saves `loss_ablation_metrics.csv` plus the comparison figure.

## 6. Where the report values come from

Use these generated files rather than manually copying numbers from the terminal:

- `outputs/eda/eda_sample_images.png`
- `outputs/eda/eda_intensity_histogram.png`
- `outputs/task1_vlm/optimized_repeated_runs.json`
- `outputs/task1_vlm/run_to_run_variability.json`
- `outputs/task2_classical/representative_regionprops.csv`
- `outputs/task2_classical/numbers_first_record.json`
- `outputs/task2_classical/numbers_first_narrative.txt`
- `outputs/task3_unet/validation_metrics_bce_dice.json`
- `outputs/task3_unet/method_comparison_metrics.csv`
- `outputs/task3_unet/otsu_vs_unet_validation.csv`
- `outputs/task3_unet/otsu_vs_unet_examples.png`
- `outputs/task3_unet/validation_predictions_bce_dice.png`
- `outputs/task3_unet/loss_curves_bce_dice.png`
- `outputs/task3_unet/dice_iou_curves_bce_dice.png`
- `outputs/task4_hybrid/hybrid_test_records.csv`
- `outputs/task4_hybrid/records/*.json`
- `outputs/task4_hybrid/narratives/*.txt`
- `outputs/extensions/loss_ablation/loss_ablation_metrics.csv` (if run)

## 7. Auditability / hallucination controls

The code intentionally separates deterministic measurements from generated language:

1. Images are segmented before quantitative reporting.
2. Region counts and areas are calculated by code, not guessed by an LLM.
3. The direct VLM prompt prohibits diagnosis and allows the literal value `uncertain`.
4. Ollama structured-output schemas are used for JSON-generating steps.
5. In the hybrid stage, `image_id`, `n_objects`, and `mean_area` are overwritten with the measured values after parsing, so an LLM cannot silently change them.
6. Raw LLM responses, feature tables, final JSON, and narratives are saved separately for auditing.

These choices make the structured data the source of truth and the narrative a human-readable interpretation layer.

## 8. Project structure

```text
biomedical_nuclei_hybrid_pipeline/
├── config.py
├── run_all.py
├── requirements.txt
├── prompts/
├── scripts/
│   ├── 00_download_dataset.py
│   ├── 01_prepare_eda.py
│   ├── 02_task1_vlm.py
│   ├── 03_task2_classical_llm.py
│   ├── 04_task3_train_unet.py
│   ├── 05_task4_hybrid_pipeline.py
│   ├── 06_extension_robustness.py
│   └── 07_extension_loss_ablation.py
├── src/
│   ├── data.py
│   ├── eda.py
│   ├── classical.py
│   ├── llm_utils.py
│   ├── unet_model.py
│   ├── train_utils.py
│   ├── plotting.py
│   ├── hybrid.py
│   └── robustness.py
├── data/
├── models/
├── outputs/
└── tests/
```

## 9. Reproducibility notes

- Random seed: `42` by default.
- Preprocessing resolution: `256×256`.
- Default U-Net loss: `BCE + Dice`.
- Default optimiser: Adam, learning rate `1e-3`.
- Default batch size: `4`.
- Best checkpoint is selected by validation Dice.
- Validation Dice and IoU are reported on thresholded binary predictions at `0.5`.
- If the dataset already contains split folders, they are respected where possible; otherwise a deterministic train/validation/test split is created.

## 10. Before submission

Run the pipeline from a fresh VS Code terminal, confirm that all report figures/values were generated by the submitted code, and commit the source, prompts, README, and small output CSV/JSON/PNG files you want the marker to inspect. Avoid committing the original downloaded dataset unless your module explicitly asks you to redistribute it.
