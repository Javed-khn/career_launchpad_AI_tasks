# Week 5 – CNN vs RNN vs LSTM on EuroSAT satellite images

Compares a CNN, a row-wise RNN and a row-wise LSTM on a small subset (5 classes x 400 images) of EuroSAT RGB, trained under identical conditions.

**Dataset (not included, 95 MB):** download `EuroSAT_RGB.zip` from https://zenodo.org/records/7711810 and extract it to `data_raw/EuroSAT_RGB/` inside this folder (one sub-folder per class).

Run `Week5_Task5_CNN_RNN_LSTM.ipynb` (or the `.py`); results land in `outputs/`. `make_report.py` / `build_docx.py` build the PDF / Word report from those outputs.

Test-set results (seed 42): CNN 97.3 % accuracy, LSTM 80.4 %, RNN 73.1 %.
