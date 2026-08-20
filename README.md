# Auditable Hybrid Nuclei Image Analysis Using VLMs, Classical Features and U-Net

## Overview

This project implements an auditable biomedical image-analysis pipeline for nuclei microscopy images. It combines a local multimodal VLM, classical image processing, a PyTorch U-Net, and a hybrid structured reporting stage.

The system is designed for educational use only and is not intended for clinical diagnosis or medical decision-making.

## Pipeline

```text
Raw image
  -> Grayscale + resize to 256x256
  -> EDA
  -> Task 1: Direct VLM description
  -> Task 2: Otsu segmentation + regionprops + numbers-first LLM
  -> Task 3: U-Net segmentation + Dice/IoU evaluation
  -> Task 4: U-Net mask -> region features -> JSON -> narrative -> CSV
```

## Dataset

Dataset source:

https://github.com/Nickolay-K/Assingnment-3-dataset

The preprocessing pipeline converts images to grayscale and resizes them to 256x256.

| Split | Images |
|---|---:|
| Training | 80 |
| Validation | 20 |
| Test | 12 |

## Main Results

| Method | Mean Dice | Mean IoU |
|---|---:|---:|
| Otsu + morphology | 0.9782 | 0.9573 |
| U-Net, Dice loss (12 epochs) | 0.9943 | 0.9886 |
| U-Net, BCE loss (12 epochs) | 0.9957 | 0.9914 |
| U-Net, BCE+Dice loss (12 epochs) | **0.9958** | **0.9916** |
| Final U-Net, BCE+Dice (20 epochs) | **0.9966** | **0.9933** |

The best final U-Net checkpoint occurred at epoch 18.

## Task 1 - Multimodal VLM Description

A representative image is analysed with `llama3.2-vision` through Ollama.

Two prompt styles are compared:

1. Naive prompt
2. Optimised structured prompt

The optimised prompt:
- avoids diagnosis
- restricts the model to visible evidence
- allows `"uncertain"`
- returns JSON
- uses the fields `modality`, `tissue_type`, `notable_features`, and `image_quality`

Repeated runs are saved to show run-to-run variability.

Outputs:

```text
outputs/task1_vlm/
```

## Task 2 - Classical Features + Numbers-First LLM

The classical pipeline applies:
1. Otsu thresholding
2. Morphological cleanup
3. Connected-component labelling
4. `regionprops_table` feature extraction

Measured features include area, eccentricity, solidity, mean intensity, object count, and foreground fraction.

For the representative image:
- Connected objects: 9
- Foreground fraction: 0.0197
- Mean object area: 143.56 pixels
- Mean eccentricity: 0.493
- Mean solidity: 0.956
- Mean object intensity: 0.262

Only measured numbers are supplied to the text LLM.

Example structured output:

```json
{
  "n_objects": 9,
  "density_class": "low",
  "shape_regularity": "irregular",
  "quality_flag": "review"
}
```

## Task 3 - U-Net Segmentation

A compact PyTorch U-Net is trained on the nuclei dataset.

Training setup:
- Image size: 256x256
- Optimizer: Adam
- Learning rate: 1e-3
- Epochs: 20
- Main loss: BCE + Dice
- Metrics: Dice and IoU

Final results:

```text
Mean Dice = 0.9966
Mean IoU  = 0.9933
```

A loss-ablation extension compares BCE, Dice, and BCE+Dice.

## Task 4 - Hybrid Pipeline

For each unseen test image:

```text
Test image
  -> U-Net mask
  -> connected components
  -> region features
  -> structured JSON
  -> narrative
```

Example record:

```json
{
  "image_id": "test_0000",
  "n_objects": 8,
  "mean_area": 190.875,
  "density_class": "low",
  "quality_flag": "review"
}
```

All test records are aggregated into a pandas DataFrame and saved as CSV.

## Project Structure

```text
biomedical-image-analysis-pipeline/
|
|-- README.md
|-- requirements.txt
|-- run_all.py
|-- .gitignore
|
|-- scripts/
|   |-- 00_download_dataset.py
|   |-- 01_prepare_eda.py
|   |-- 02_task1_vlm.py
|   |-- 03_task2_classical_llm.py
|   |-- 04_task3_train_unet.py
|   |-- 05_task4_hybrid_pipeline.py
|   `-- 07_extension_loss_ablation.py
|
|-- src/
|-- prompts/
`-- outputs/
    |-- eda/
    |-- task1_vlm/
    |-- task2_classical/
    |-- task3_unet/
    |-- task4_hybrid/
    `-- extensions/
```

## Installation

### Clone

```bash
git clone https://github.com/yashs14282001/biomedical-image-analysis-pipeline.git
cd biomedical-image-analysis-pipeline
```

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ollama Setup

Install Ollama and pull the required models:

```powershell
ollama pull llama3.2-vision
ollama pull llama3.2:3b
```

Check installed models:

```powershell
ollama list
```

Test the vision model:

```powershell
ollama run llama3.2-vision
```

`llama3.2-vision` requires substantial RAM. Around 16 GB system memory or more is recommended.

## Running the Project

Run the tasks in order:

```powershell
python scripts/00_download_dataset.py
python scripts/01_prepare_eda.py
python scripts/02_task1_vlm.py
python scripts/03_task2_classical_llm.py
python scripts/04_task3_train_unet.py --epochs 20 --loss bce_dice
python scripts/05_task4_hybrid_pipeline.py
```

Optional loss ablation:

```powershell
python scripts/07_extension_loss_ablation.py --epochs 12
```

Or run the main workflow with:

```powershell
python run_all.py --epochs 20
```

## Output Folders

```text
outputs/eda/
outputs/task1_vlm/
outputs/task2_classical/
outputs/task3_unet/
outputs/task4_hybrid/
outputs/extensions/
```

These folders contain the figures, metrics, JSON records, narratives, model comparisons, and aggregated CSV used in the report.

## Auditability and Hallucination Control

The pipeline reduces hallucination risk by:
- explicitly prohibiting diagnosis in prompts
- allowing `"uncertain"` when evidence is insufficient
- preferring structured JSON to unrestricted free text
- passing only measured numbers to the Task 2 LLM
- retaining measured values separately from generated interpretations
- treating deterministic numerical fields as the source of truth
- generating narratives only after structured records are created

These safeguards reduce, but do not eliminate, hallucination risk.

## Limitations

- Small and homogeneous dataset
- Internal validation only
- High Dice/IoU does not guarantee external generalisation
- VLM responses vary between runs
- JSON formatting does not guarantee semantic correctness
- No clinical validation has been performed

## Clinical Use Disclaimer

This project is for educational and research purposes only.

It is not a medical device, is not clinically validated, and must not be used for diagnosis, treatment, patient management, or other clinical decision-making.

## References

- Otsu, N. (1979). *A Threshold Selection Method from Gray-Level Histograms*. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.
- Sudre, C. H., Li, W., Vercauteren, T., Ourselin, S., & Cardoso, M. J. (2017). *Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations*.
- Ollama documentation: https://docs.ollama.com/

## Author

**Yash Salve**  
MSc Data Science  
University of Hertfordshire
