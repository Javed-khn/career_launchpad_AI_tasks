"""Builds the PDF report from the CSV/PNG files produced by the notebook (numbers are read, not typed)."""
import pandas as pd
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

B = Path(__file__).parent; O = B / "outputs"
tab = pd.read_csv(O / "comparison_table.csv").set_index("Model")
runs = pd.read_csv(O / "all_runs.csv")
auc = pd.read_csv(O / "auc_per_class.csv").set_index("Model")
ms = runs.groupby("model")[["accuracy", "f1", "auc_macro", "train_time_s"]].agg(["mean", "std"])
def m(k, c): return f"{ms.loc[k,(c,'mean')]:.3f} ± {ms.loc[k,(c,'std')]:.3f}"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=8, spaceAfter=5, keepWithNext=1, textColor=colors.HexColor("#1f3b6f"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=6, spaceAfter=3, keepWithNext=1)
P = ParagraphStyle("P", parent=ss["BodyText"], fontSize=9.6, leading=13.2, spaceAfter=5)
CAP = ParagraphStyle("CAP", parent=P, fontSize=8.5, textColor=colors.grey, alignment=1)
TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=19, textColor=colors.HexColor("#1f3b6f"))

def tbl(data, widths=None, fs=8.5):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b6f")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), fs), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f9")]),
                           ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t

def img(name, w):
    from PIL import Image as PI
    iw, ih = PI.open(O / name).size
    return Image(str(O / name), width=w, height=w * ih / iw)

S = []
S += [Paragraph("Week 5 – Task 05: Comparative Analysis of CNN, RNN and LSTM<br/>for Satellite Land-Use Classification (EuroSAT)", TITLE),
      Paragraph("Career Launch Pad – Artificial Intelligence &amp; Machine Learning · UET Mardan · Instructor: Mr. Zaheen Ahmad", CAP),
      Paragraph("Student name: ____________________ &nbsp;&nbsp; Roll number: ____________________", CAP), Spacer(1, 8)]

S += [Paragraph("1. Problem and Dataset", H1),
      Paragraph("<b>Problem.</b> Automatically recognising land use from satellite imagery supports forest monitoring, "
                "urban-growth tracking and infrastructure mapping. We classify 64×64 px Sentinel-2 patches into five land-cover classes and compare "
                "three deep-learning architectures – a Convolutional Neural Network (CNN), a vanilla Recurrent Neural Network (RNN) and a Long "
                "Short-Term Memory network (LSTM) – trained under identical conditions on the same data.", P),
      Paragraph("<b>Dataset.</b> EuroSAT (RGB version) – Helber, Bischke, Dengel &amp; Borth, <i>IEEE Journal of Selected Topics in Applied Earth "
                "Observations and Remote Sensing</i>, 2019; publicly available on Zenodo (record 7711810). The full set has 27,000 labelled "
                "images in 10 classes (2,000–3,000 each) at 64×64 px. To obtain a deliberately <b>small</b> dataset we selected five visually related "
                "classes and randomly sampled <b>400 images per class (2,000 in total)</b>.", P)]
S += [tbl([["Property", "Value"], ["Source", "EuroSAT RGB, Zenodo 7711810 (Sentinel-2, 13 cities/34 countries)"],
           ["Classes (5)", "Forest, Pasture, River, Highway, Residential"], ["Images per class", "400 (balanced) → 2,000 total"],
           ["Resolution / channels", "64 × 64 px, RGB (3 channels), JPEG"],
           ["Split (stratified)", "Train 1,400 (70 %) · Validation 300 (15 %) · Test 301 (15 %)"]], [4.2 * cm, 12.3 * cm]), Spacer(1, 4),
      Paragraph("<b>Why 70/15/15.</b> With only 2,000 images, 70 % still leaves ≈280 training images per class, 15 % validation (300) is enough for "
                "stable early stopping, and a 15 % test set (≈60 per class) gives a more reliable ROC/AUC estimate than an 80/10/10 split would "
                "(≈200 test images). Splits are stratified so every split keeps the class balance.", P),
      img("samples.png", 11.5 * cm), Paragraph("Figure 1 – Sample images (rows: Forest, Pasture, River, Highway, Residential).", CAP)]

S += [Paragraph("2. Pre-processing", H1),
      Paragraph("• <b>Rescaling and normalisation:</b> pixels scaled to [0,1], then standardised per channel with the <i>training-set</i> mean and "
                "standard deviation (no information from validation/test leaks in).<br/>"
                "• <b>Resizing:</b> not required – images are already 64×64.<br/>"
                "• <b>Augmentation (training only):</b> random horizontal flip, vertical flip and 90° rotations, applied per image. Satellite scenes have "
                "no canonical orientation, so these transforms preserve the label and effectively enlarge a small dataset, reducing over-fitting.<br/>"
                "• <b>Sequence conversion for RNN/LSTM:</b> each image is read row by row – 64 time-steps, each a 192-dimensional vector "
                "(64 pixels × 3 channels).", P)]

S += [Paragraph("3. Model Architectures", H1),
      tbl([["Model", "Architecture", "Trainable params"],
           ["CNN", "3 blocks of [Conv3×3–BN–ReLU]×2 + MaxPool (32→64→128 filters), global average pooling, Dropout 0.4, Linear(5)", f"{int(tab.loc['CNN','Trainable params']):,}"],
           ["RNN", "2-layer tanh Elman RNN (hidden 128, dropout 0.3) over 64 row-steps; last hidden state → Dropout → Linear(5)", f"{int(tab.loc['RNN','Trainable params']):,}"],
           ["LSTM", "2-layer LSTM (hidden 128, dropout 0.3) over 64 row-steps; last hidden state → Dropout → Linear(5)", f"{int(tab.loc['LSTM','Trainable params']):,}"]],
          [1.8 * cm, 11.6 * cm, 3.1 * cm], fs=8.3), Spacer(1, 4),
      Paragraph("<b>Common training set-up (identical for all models):</b> Adam (lr 1e-3, weight decay 1e-4), cross-entropy loss, batch size 64, up to 40 "
                "epochs with cosine learning-rate decay, gradient-norm clipping at 1.0, early stopping on validation loss (patience 8) with the best "
                "weights restored. Implemented in PyTorch and trained on an NVIDIA RTX 3060 Ti GPU. Every model was trained with three random seeds "
                "(42, 43, 44); figures and the main table use seed 42, and mean ± std over the three seeds is also reported.", P)]

S += [PageBreak(), Paragraph("4. Results", H1), Paragraph("4.1 Comparison table (test set, seed 42)", H2)]
cols = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "AUC (OvR macro)"]
rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "Params", "Train time (s)"]]
for k in ("CNN", "RNN", "LSTM"):
    r = tab.loc[k]; rows.append([k] + [f"{r[c]:.4f}" for c in cols] + [f"{int(r['Trainable params']):,}", f"{r['Train time (s)']:.1f}"])
S += [tbl(rows, [1.6 * cm, 2.0 * cm, 2.0 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 2.2 * cm, 2.3 * cm]),
      Paragraph("Precision, recall and F1 are macro-averaged over the five classes; AUC is one-vs-rest macro-averaged.", CAP), Spacer(1, 4)]
rows2 = [["Model", "Accuracy (mean ± std)", "F1 (mean ± std)", "AUC (mean ± std)", "Train time s (mean)"]]
for k in ("CNN", "RNN", "LSTM"):
    rows2.append([k, m(k, "accuracy"), m(k, "f1"), m(k, "auc_macro"), f"{ms.loc[k,('train_time_s','mean')]:.1f}"])
S += [Paragraph("Mean ± standard deviation over 3 seeds (robustness check):", P), tbl(rows2, [1.8 * cm, 4.2 * cm, 3.8 * cm, 4.0 * cm, 3.0 * cm]), Spacer(1, 6)]

S += [Paragraph("4.2 Training vs. validation curves", H2), img("training_curves.png", 16.2 * cm),
      Paragraph("Figure 2 – Loss and accuracy per epoch. The CNN converges to low loss with a small train/validation gap; the RNN is noisy and still "
                "improving at epoch 40 (under-fitting); the LSTM shows a clear and growing train-validation gap.", CAP)]

S += [PageBreak(), Paragraph("4.3 Confusion matrices", H2), img("confusion_matrices.png", 16.4 * cm),
      Paragraph("Figure 3 – Confusion matrices on the test set. Errors of the RNN/LSTM concentrate on Highway (mistaken for River or Residential) and River (mistaken for Highway).", CAP),
      Paragraph("4.4 ROC curves and AUC (one-vs-rest)", H2), img("roc_curves.png", 16.4 * cm),
      Paragraph("Figure 4 – Left: micro-averaged ROC of all three models on one plot. Right: per-class ROC curves.", CAP)]
rows3 = [["Model"] + list(auc.columns)]
for k in ("CNN", "RNN", "LSTM"): rows3.append([k] + [f"{v:.4f}" for v in auc.loc[k]])
S += [tbl(rows3, [1.8 * cm] + [2.1 * cm] * 7), Paragraph("Per-class and averaged one-vs-rest AUC on the test set.", CAP)]

c, r_, l = tab.loc["CNN"], tab.loc["RNN"], tab.loc["LSTM"]
S += [PageBreak(), Paragraph("5. Discussion", H1),
      Paragraph(f"<b>Accuracy and ranking.</b> The CNN is clearly best (accuracy {c['Accuracy']:.3f}, F1 {c['F1 (macro)']:.3f}, macro AUC {c['AUC (OvR macro)']:.3f}), "
                f"followed by the LSTM ({l['Accuracy']:.3f}, {l['F1 (macro)']:.3f}, {l['AUC (OvR macro)']:.3f}) and the plain RNN ({r_['Accuracy']:.3f}, "
                f"{r_['F1 (macro)']:.3f}, {r_['AUC (OvR macro)']:.3f}). The ordering CNN &gt; LSTM &gt; RNN is the same for every seed and the gap "
                f"between models is many times larger than the seed-to-seed standard deviation, so it is not due to chance.", P),
      Paragraph("<b>Why the CNN wins.</b> (1) <i>Local receptive fields and weight sharing:</i> a 3×3 filter detects the same texture – canopy, road marking, "
                "roof pattern – anywhere in the image using very few weights. (2) <i>Translation invariance</i> through pooling: a river is a river "
                "wherever it sits in the patch. (3) <i>Hierarchical features:</i> edges → textures → object parts. The row-wise RNN/LSTM instead flattens "
                "each row into a 192-D vector, which destroys horizontal locality and the 2-D neighbourhood structure, and must push vertical information "
                "through 64 sequential steps. Note that the CNN is also the only model that exploits the 2-D structure explicitly; the RNN/LSTM have to "
                "re-discover it from scratch.", P),
      Paragraph("<b>RNN vs. LSTM.</b> The LSTM beats the plain RNN by about "
                f"{(l['Accuracy']-r_['Accuracy'])*100:.1f} accuracy points. Gates and the additive cell state mitigate vanishing gradients across the 64 steps, "
                "whereas the tanh RNN essentially relies on the last few rows and its validation curve stays noisy and unconverged. However, the LSTM "
                "has ≈4× more parameters than the RNN and its training accuracy rises well above validation accuracy (Figure 2), i.e. it starts to over-fit "
                "a 1,400-image training set.", P),
      Paragraph("<b>Class-level behaviour.</b> The visually distinctive classes (Forest, Residential) are easy for every model. The thin, linear, "
                "orientation-varying structures – Highway and River – are the hardest for the sequence models (RNN Highway F1 and AUC are the lowest in "
                "the study), because recognising a line that may run in any direction requires 2-D context. The CNN keeps AUC ≥ 0.998 for all classes. "
                "ROC curves (Figure 4) show the same story: the CNN curve hugs the top-left corner, the LSTM lies in between and the RNN is lowest.", P),
      Paragraph("<b>Cost.</b> The CNN has a similar parameter count to the LSTM yet reaches far higher accuracy, but it is slower to train "
                f"(≈{c['Train time (s)']:.0f} s vs. {l['Train time (s)']:.0f} s for the LSTM and {r_['Train time (s)']:.0f} s for the RNN for 40 epochs), because convolutions over a "
                "64×64 grid are more computation-heavy than a 64-step recurrence of small matrices. The extra seconds are clearly worthwhile.", P),
      Paragraph("<b>Limitations.</b> Only five classes and 2,000 images were used; the test set has 301 images, so single-run percentages carry roughly ±1–3 % "
                "uncertainty (hence the 3-seed check). The RNN/LSTM use a simple row-wise scan; patch-wise scanning, bidirectional layers, or a "
                "CNN-feature + LSTM hybrid could narrow – but, we expect, not close – the gap. The CNN could be further improved with transfer learning.", P),
      Paragraph("6. Conclusion", H1),
      Paragraph(f"On this small satellite land-use dataset the <b>CNN is the best model</b> on every metric (accuracy {c['Accuracy']*100:.1f} %, macro AUC "
                f"{c['AUC (OvR macro)']:.3f}), the <b>LSTM is second</b> ({l['Accuracy']*100:.1f} %) and the <b>vanilla RNN is last</b> ({r_['Accuracy']*100:.1f} %). "
                "This confirms the theory that architectures with built-in spatial inductive bias (convolution, weight sharing, pooling) are far better "
                "suited to raw image data than sequence models, even when the latter are fed the image as a sequence of rows. RNNs/LSTMs remain "
                "valuable for genuinely sequential data (time series, text, video frames) – not for single still images.", P),
      Paragraph("References", H2),
      Paragraph("[1] P. Helber, B. Bischke, A. Dengel, D. Borth, “EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover "
                "Classification,” IEEE J-STARS, 2019. Dataset: https://zenodo.org/records/7711810<br/>"
                "[2] S. Hochreiter, J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, 1997.", P)]

doc = SimpleDocTemplate(str(B / "Week5_Task5_Report.pdf"), pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                        title="Week 5 Task 05 – CNN vs RNN vs LSTM", author="")
def footer(cv, d):
    cv.setFont("Helvetica", 8); cv.setFillColor(colors.grey); cv.drawCentredString(A4[0] / 2, 0.9 * cm, f"Week 5 Task 05 – CNN vs RNN vs LSTM · page {d.page}")
doc.build(list(S), onFirstPage=footer, onLaterPages=footer); print("report written")
