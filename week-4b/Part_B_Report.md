# Week 4 – Part B: Deep Learning Architectures — Summary Report

**Framework:** PyTorch 2.14 (CUDA 12.6) · **Hardware:** NVIDIA RTX 3060 Ti · **Seed:** 42 for all models

| Notebook | Task | Dataset |
|---|---|---|
| `01_ANN_Tabular.ipynb` | B1 – ANN, tabular binary classification | Pima Indians Diabetes (768 rows, 8 features) |
| `02_CNN_FashionMNIST.ipynb` | B2 – CNN, 10-class image classification | Fashion-MNIST (60k train / 10k test, 28×28) |
| `03_RNN_IMDB.ipynb` | B3 – SimpleRNN vs LSTM vs BiLSTM, sentiment | IMDB reviews (25k train / 25k test) |
| `04_YOLO_Detection.ipynb` | B4 – pretrained YOLOv8n object detection | 8 images + 75-frame video |

All figures (training curves, confusion matrices, annotated detections) are saved in `outputs/`, and the metrics in `outputs/results.json`.

---

## 1. Results (held-out test sets)

| Model | Accuracy | Precision | Recall | F1 | Train time | Trainable params |
|---|---|---|---|---|---|---|
| **ANN** (32→16, LeakyReLU, dropout 0.4) | 0.750 | 0.676 | 0.561 | 0.613 | ~1 s | 850 |
| **CNN** – no augmentation | **0.937** | 0.937 | 0.937 | 0.937 | 109 s (15 ep) | 585 k |
| **CNN** – with augmentation | 0.923 | 0.923 | 0.923 | 0.923 | 199 s (15 ep) | 585 k |
| **SimpleRNN** | 0.736 | 0.760 | 0.690 | 0.723 | 8 s (8 ep) | 2.59 M |
| **LSTM** | 0.847 | 0.823 | 0.885 | 0.853 | 15 s (8 ep) | 2.69 M |
| **BiLSTM** | **0.869** | 0.860 | 0.883 | 0.871 | 23 s (8 ep) | 2.82 M |

ANN precision/recall are for the positive (diabetic) class. CNN values are macro-averaged over 10 classes.
The RNN parameter counts are dominated by the 20k×128 embedding (2.56 M). The recurrent layers themselves hold 33 k (RNN), 132 k (LSTM) and 264 k (BiLSTM) parameters.

| YOLOv8n (pretrained, COCO) | Value |
|---|---|
| Images / total detections (conf ≥ 0.25) | 8 / 42 |
| Mean confidence of kept boxes | 0.60 |
| Inference per image (GPU, after warm-up) | 3.3 – 14.7 ms (mean 8.9 ms) |
| Video: 75 frames, 640×480 | 3.4 ms/frame ≈ 297 FPS model throughput |
| Detections at conf 0.10 / 0.25 / 0.50 / 0.75 | 91 / 42 / 25 / 16 |

---

## 2. Per-model findings

**ANN (B1).** I zeroed impossible values (glucose, BMI, insulin…) and imputed them with *train-set* medians, then standardised the features. A grid of 27 configurations (3 depths × 3 activations × 3 dropout rates) was ranked by validation loss. The best configuration was a medium-sized network with heavy dropout (0.4): with only 537 training rows, regularisation mattered more than depth. Accuracy is 75%, but **recall of 0.56 means almost half of the diabetic patients are missed**. For a screening use case, the decision threshold or a class-weighted loss should be tuned toward recall. The test set has only 116 rows, so ±1 sample moves accuracy by ~0.9 points.

**CNN (B2).** The model has three Conv–BN–ReLU blocks (32/64/128 filters) with max-pooling and increasing dropout. Without augmentation it begins to over-fit after about epoch 6: training accuracy rises to ~95.5% while validation plateaus at ~93.7%. With augmentation (random crop, ±10° rotation, horizontal flip) it shows no over-fitting, since validation stays above training, and both curves are *still rising* at epoch 15. Under this fixed 15-epoch budget the augmented model scored 1.4 points lower. Augmentation is a regulariser that trades fitting speed for generalisation. Fashion-MNIST images are centred and upright, so rotation adds variation the test set never contains. Most errors are among shirt, T-shirt, pullover and coat.

**RNN / LSTM (B3).** The three models are identical except for the recurrent cell, so the comparison isolates gating. The SimpleRNN trained erratically (validation accuracy moved between 57% and 74%), a symptom of vanishing gradients over 256-step sequences. Gradient clipping limits exploding gradients but cannot fix vanishing ones. The LSTM's gated, additive cell-state update added **+11 points** of accuracy for ~1.8× the training time. The BiLSTM added another 2 points but started to over-fit after epoch 5 (validation loss rose while training loss fell). The best-validation weights were kept.

**YOLO (B4).** YOLOv8n correctly found the bus and pedestrians, two cats and remotes, a stop sign, the bed and furniture, a skier and skis, and two people with a tie. The errors are informative:
- A "person" at 0.71 in the living-room image appears to be a false positive.
- A distant second "stop sign" at 0.35 and a "truck" at 0.30 are low-confidence guesses.
- A "sports ball" at 0.25 sits right at the threshold.

Raising the threshold from 0.25 to 0.5 removes 40% of the boxes, mostly these weak ones. On the video, detections stay stable from frame to frame, at about 14 objects per frame.

---

## 3. Architecture comparison

| | ANN | CNN | RNN / LSTM | YOLO |
|---|---|---|---|---|
| **Input structure** | Fixed-length feature vector, no order | 2-D grid, local spatial patterns | Ordered sequence, variable length | Full image, many objects at once |
| **Key inductive bias** | None, every input connects to every unit | Locality + weight sharing + translation invariance | Shared weights over time + memory state | Grid-based regression of boxes + classes in one pass |
| **Complexity (this work)** | 850 params | 585 k | 2.6–2.8 M (mostly embedding) | ~3.2 M (nano), pretrained |
| **Training time (this work)** | ~1 s | 2–3.5 min | 8–23 s | None (inference only) |
| **Data needs** | Low; over-fits easily | Moderate to high; benefits from augmentation | High for text | Huge, but transferred via pretraining |
| **Main weakness seen** | Low recall on small, imbalanced data | Over-fits without regularisation | Vanilla RNN: vanishing gradients | Small or ambiguous objects, threshold-sensitive |
| **Best fit** | Tabular / structured records | Image classification, spatial signals | Text, time series, sensor streams | Real-time localisation + classification |

**Conclusion.** Each architecture worked best on the data structure it was designed for.
- The **ANN** is the right tool for tabular data. It is tiny and trains in seconds, but small datasets limit it, and gradient-boosted trees remain a strong alternative there.
- The **CNN**'s convolutions exploit spatial locality to reach ~94% accuracy on images with a modest parameter count.
- For sequences, **gating is essential**: the LSTM beat the SimpleRNN by 11 points under identical settings. Bidirectionality helps when the whole sequence is available at prediction time.
- **YOLO** shows the value of transfer learning. A pretrained 3 M-parameter model gives usable multi-object detection at hundreds of frames per second with no training. Its output still needs interpretation, because the confidence threshold directly trades false positives against missed objects.

---

## 4. Reproducing

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python build_notebooks.py          # regenerates the four notebooks (optional; they are included)
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```
The datasets (OpenML diabetes, Fashion-MNIST, IMDB), the YOLO weights and the sample images download automatically on the first run. The notebooks run on a CPU too, just more slowly.

**Limitations.** Each model was trained once with a single seed, so differences under about 1 point (for example CNN with vs. without augmentation) are within run-to-run noise. Training budgets are short and were not equalised for convergence. The YOLO video is a synthetic pan over a still image, not real footage; set `VIDEO = "your_clip.mp4"` or `source=0` (webcam) in notebook 04 to use real video.
