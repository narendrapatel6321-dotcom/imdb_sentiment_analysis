# IMDB Sentiment Analysis — Dense vs CNN vs LSTM vs BiLSTM (Base & GloVe-Enhanced)

A structured comparison of four neural network architectures on the IMDB movie review sentiment dataset, built with TensorFlow/Keras. The project runs two full rounds of experiments — base models with learned embeddings and enhanced models initialized with pretrained GloVe vectors — with thorough evaluation and visualization at each stage.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Models](#models)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Visualizations](#visualizations)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

---

## Project Overview

The goal of this project is to benchmark four progressively complex architectures on binary sentiment classification (Positive / Negative) using the IMDB 50K Movie Reviews dataset. Each architecture is trained twice — once with randomly initialized embeddings (base) and once initialized with pretrained GloVe 6B 100d vectors (enhanced).

| Model           | Version   | Embedding         | Key Features                                              |
|-----------------|-----------|-------------------|-----------------------------------------------------------|
| Dense           | Base      | Learned           | GlobalAveragePooling over embeddings. Fast baseline.      |
| CNN             | Base      | Learned           | Two Conv1D layers, BatchNorm, L2 regularization.          |
| LSTM            | Base      | Learned           | Single unidirectional LSTM.                               |
| BiLSTM          | Base      | Learned           | Stacked Bidirectional LSTM.                               |
| Dense           | GloVe     | GloVe 6B 100d     | Same architecture, pretrained embedding init.             |
| CNN             | GloVe     | GloVe 6B 100d     | Parallel multi-kernel Conv1D (3, 4, 5), Functional API.   |
| LSTM            | GloVe     | GloVe 6B 100d     | Single LSTM with GloVe initialization.                    |
| BiLSTM          | GloVe     | GloVe 6B 100d     | Stacked BiLSTM with custom **Attention Pooling**.         |

All models use the **Adam optimizer** and **binary cross-entropy loss**, with early stopping, learning rate scheduling, and CSV logging.

Global random seed: `21` (Python, NumPy, TensorFlow)


---

## Repository Structure

```
├── Sentiment_Analysis_IMDB.ipynb       # Main notebook
├── helper.py                           # Utility functions (plot_training_curve, evaluate_model)
│
├── plots/                              # Training curves and confusion matrices (16 plots total)
│
└── README.md
```

> **Trained Models & Training Logs:** Hosted on Google Drive due to GitHub's file size limits.
>  [Download Models & Training Logs](https://drive.google.com/drive/folders/1UBXGualLQ5LbvVndOghFGrHD4UD73qp4?usp=drive_link)

---

## Dataset

**IMDB 50K Movie Reviews** — downloaded automatically via `kagglehub`.

| Split      | Samples |
|------------|---------|
| Train      | 40,000  |
| Validation | 5,000   |
| Test       | 5,000   |

- Split ratio: 80% train / 10% validation / 10% test — stratified on label
- Labels: `positive → 1`, `negative → 0`
- Batch size: `64` for all splits

### Text Preprocessing

Raw reviews are cleaned before being passed to the `TextVectorization` layer using the `remove_unwanted()` function:

- Strip HTML tags (via BeautifulSoup)
- Remove URLs
- Remove digits
- Normalize whitespace

The `TextVectorization` layer then handles lowercasing and punctuation removal.

### Vectorization & Embedding

| Parameter       | Value   |
|-----------------|---------|
| Vocabulary size | 20,000  |
| Sequence length | 300     |
| Embedding dim (base) | 128 |
| Embedding dim (GloVe) | 100 |

---

## Models

### Round 1 — Base Models (Learned Embeddings)

All base models share the same input pipeline (`TextVectorization → Embedding`) and classification head (`Dense(64, relu) → Dropout(0.4) → Dense(1, sigmoid)`).

---

#### 1. Dense

```
Input (string)
  └── TextVectorization
  └── Embedding(vocab, 128)
  └── GlobalAveragePooling1D
  └── Dense(64, relu, L2) → Dropout(0.4)
  └── Dense(1, sigmoid)
```

- Fastest baseline — no sequential modeling
- Treats reviews as bags of embedded words

---

#### 2. CNN

```
Input (string)
  └── TextVectorization
  └── Embedding(vocab, 128)
  └── SpatialDropout1D(0.3)
  └── Conv1D(128, kernel=5, relu, L2) → BatchNorm → Dropout(0.4)
  └── Conv1D(64, kernel=3, relu, L2) → Dropout(0.4)
  └── GlobalMaxPooling1D
  └── Dense(64, relu, L2) → Dropout(0.4)
  └── Dense(1, sigmoid)
```

---

#### 3. LSTM

```
Input (string)
  └── TextVectorization
  └── Embedding(vocab, 128, mask_zero=True)
  └── SpatialDropout1D(0.3)
  └── LSTM(128, dropout=0.3)
  └── Dense(64, relu, L2) → Dropout(0.4)
  └── Dense(1, sigmoid)
```

---

#### 4. BiLSTM

```
Input (string)
  └── TextVectorization
  └── Embedding(vocab, 128, mask_zero=True)
  └── SpatialDropout1D(0.3)
  └── Bidirectional(LSTM(128, dropout=0.3, return_sequences=True))
  └── Bidirectional(LSTM(64, dropout=0.3))
  └── Dense(64, relu, L2) → Dropout(0.4)
  └── Dense(1, sigmoid)
```

---

### Round 2 — Enhanced Models (GloVe Embeddings)

Built with `build_model_v2()`. Embeddings are initialized from **GloVe 6B 100d** vectors and fine-tuned during training. Words not found in GloVe receive small random vectors (scale=0.01) rather than zeros.

Key upgrades over the base round:

- **Dense**: Same architecture, GloVe embedding init.
- **CNN**: Upgraded to a **parallel multi-kernel** architecture (Functional API) with three Conv1D branches (kernel sizes 3, 4, 5) concatenated before the head.
- **LSTM**: Same architecture, GloVe embedding init.
- **BiLSTM**: Stacked BiLSTM with a custom **`_AttentionPooling`** layer replacing the final hidden state — learns a scalar importance score per timestep and returns a weighted sum, better suited for long reviews.

---

## Training Configuration

| Parameter               | All Models         |
|-------------------------|--------------------|
| Batch Size              | 64                 |
| Max Epochs              | 40                 |
| Learning Rate (base)    | 1e-4               |
| Learning Rate (GloVe)   | 3e-4               |
| Early Stopping Patience | 6 epochs           |
| ReduceLROnPlateau       | ✓ (factor=0.2, patience=4, min_lr=1e-6) |
| Loss Function           | Binary Crossentropy |
| Optimizer               | Adam               |
| CSV Logging             | ✓ (per experiment) |
| Best Weights Restored   | ✓                  |

---

## Results

### Test Set Performance

| Model         | Version | Test Accuracy | Test Loss | Weighted F1 |
|---------------|---------|---------------|-----------|-------------|
| Dense         | Base    | 89.80%        | 0.2753    | 0.90        |
| CNN           | Base    | 88.36%        | 0.3155    | 0.88        |
| LSTM          | Base    | 89.42%        | 0.2661    | 0.89        |
| BiLSTM        | Base    | 90.00%        | 0.2546    | 0.90        |
| Dense         | GloVe   | 89.60%        | 0.2701    | 0.90        |
| CNN           | GloVe   | 89.36%        | 0.2642    | 0.89        |
| **LSTM**      | **GloVe** | **91.32%**  | **0.2265**| **0.91**    |
| BiLSTM        | GloVe   | 91.12%        | 0.2298    | 0.91        |

> **Best model: LSTM + GloVe** — 91.32% accuracy, 0.2265 loss, 0.91 weighted F1.

### Per-Class F1 Score (Best Model — LSTM + GloVe)

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negative | 0.92      | 0.91   | 0.91     | 2,500   |
| Positive | 0.91      | 0.92   | 0.91     | 2,500   |

---

## Visualizations

Training curves are plotted using `plot_training_curve()` from `helper.py`, which generates a two-panel figure (loss + accuracy) with a marker at the best validation loss epoch.

Evaluation plots are generated using `evaluate_model()` from `helper.py`, which prints test metrics and renders a confusion matrix heatmap.

### Training Curves

Each plot shows train vs. validation loss (top) and accuracy (bottom), with a marker at the best validation loss epoch.

#### Dense — Base
![Dense Base Training Curve](plots/training_curve_dense_base.png)

#### CNN — Base
![CNN Base Training Curve](plots/training_curve_cnn_base.png)

#### LSTM — Base
![LSTM Base Training Curve](plots/training_curve_lstm_base.png)

#### BiLSTM — Base
![BiLSTM Base Training Curve](plots/training_curve_bilstm_base.png)

#### Dense — GloVe
![Dense GloVe Training Curve](plots/training_curve_dense_glove.png)

#### CNN — GloVe
![CNN GloVe Training Curve](plots/training_curve_cnn_glove.png)

#### LSTM — GloVe
![LSTM GloVe Training Curve](plots/training_curve_lstm_glove.png)

#### BiLSTM — GloVe
![BiLSTM GloVe Training Curve](plots/training_curve_bilstm_glove.png)

---

### Confusion Matrices

Rows = True label, Columns = Predicted label. Classes: Negative (0), Positive (1).

#### Dense — Base
![Dense Base Confusion Matrix](plots/confusion_matrix_dense_base.png)

#### CNN — Base
![CNN Base Confusion Matrix](plots/confusion_matrix_cnn_base.png)

#### LSTM — Base
![LSTM Base Confusion Matrix](plots/confusion_matrix_lstm_base.png)

#### BiLSTM — Base
![BiLSTM Base Confusion Matrix](plots/confusion_matrix_bilstm_base.png)

#### Dense — GloVe
![Dense GloVe Confusion Matrix](plots/confusion_matrix_dense_glove.png)

#### CNN — GloVe
![CNN GloVe Confusion Matrix](plots/confusion_matrix_cnn_glove.png)

#### LSTM — GloVe
![LSTM GloVe Confusion Matrix](plots/confusion_matrix_lstm_glove.png)

#### BiLSTM — GloVe
![BiLSTM GloVe Confusion Matrix](plots/confusion_matrix_bilstm_glove.png)

---

## Requirements

```
tensorflow >= 2.x
numpy
pandas
matplotlib
seaborn
scikit-learn
beautifulsoup4
kagglehub
```

Install with:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn beautifulsoup4 kagglehub
```

GloVe embeddings (`glove.6B.zip`, ~822MB) are downloaded automatically from the Stanford NLP servers and extracted to Google Drive on first run.

---

## How to Run

1. Open the notebook in Google Colab and mount your Google Drive (required for GloVe storage):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Upload `helper.py` to your Colab session or place it in the same Drive folder as the notebook.

3. Run all cells in order. The notebook is organized into self-contained sections:
   - **Section 1** — Imports
   - **Section 2** — Reproducibility (seed setup)
   - **Section 3** — Data preparation & utilities (cleaning, splitting, model builders)
   - **Section 4** — Training & evaluation: Base models (Dense, CNN, LSTM, BiLSTM)
   - **Section 5** — Enhanced model utilities (GloVe loaders, Attention layer, multi-kernel CNN)
   - **Section 6** — Training & evaluation: GloVe-enhanced models

4. The IMDB dataset is downloaded automatically via `kagglehub` on first run.

5. Trained models are saved as `.keras` files and training logs as `.csv` files in the working directory.

> The notebook was developed on Google Colab with a **T4 GPU**. GPU acceleration is strongly recommended, especially for the LSTM and BiLSTM models.
