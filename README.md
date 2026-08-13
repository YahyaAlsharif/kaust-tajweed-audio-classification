# Tajweed Audio Classification — KAUST Academy Challenge 4

A 4-class audio classification solution developed for **KAUST Academy AI Summer School 2026 — Kaggle Challenge 4**.

The task was to classify Tajweed rules from Quranic recitation audio into four classes:

* `Idgham`
* `Ikhfa`
* `Iqlab`
* `Izhar`

The final model used a **ConvNeXt-Tiny** backbone with a three-channel time-frequency representation built from **log-mel spectrograms, temporal deltas, and delta-deltas**.

## Result

| Metric                       |            Result |
| ---------------------------- | ----------------: |
| OOF Macro-F1                 |        **0.9438** |
| Private Leaderboard Macro-F1 |        **0.9608** |
| Public Leaderboard Macro-F1  |            0.5371 |
| Final Rank                   |          **17th** |
| Training Data                | 1,515 audio clips |
| Test Data                    |   747 audio clips |

The public leaderboard represented only a small portion of the test set, so model selection relied primarily on frozen cross-validation rather than public leaderboard performance.

![Final leaderboard](images/leaderboard.png)

## Best Model

The strongest solution used:

```text
Audio
  ↓
16 kHz resampling
  ↓
4-second center-focused crop
  ↓
Log-Mel Spectrogram
  ├── Log-Mel
  ├── Temporal Delta
  └── Temporal Delta-Delta
  ↓
3 × 128 × 401 representation
  ↓
ConvNeXt-Tiny
  ↓
4-class Tajweed prediction
```

The backbone used **supervised ImageNet-1K pretrained weights**. No audio-specific, Quran-specific, or Tajweed-specific pretrained models were used.

## Why ConvNeXt + Delta Features?

The initial experiments showed that simply increasing model capacity was not enough.

A larger EfficientNetV2-S improved cross-validation relative to the early baseline, but was still clearly behind the final solution. The largest improvement came from changing **how the audio was represented**.

Instead of giving the model only a log-mel spectrogram, the final system used three meaningful channels:

1. **Log-Mel** — spectral energy.
2. **Delta** — first-order temporal changes.
3. **Delta-Delta** — second-order temporal changes.

This gives the image backbone direct information about both spectral structure and how pronunciation evolves over time.

ConvNeXt-Tiny then provided a strong and efficient ImageNet-pretrained convolutional backbone for learning from this structured 2D representation.

## Development Path

The solution was reached through several iterations rather than selecting the final architecture immediately.

### 1. EfficientNet-B0 Baseline

The first strong baseline used:

```text
Audio → Log-Mel → ImageNet EfficientNet-B0
```

It produced high cross-validation scores, but the validation protocol required further investigation because the small public leaderboard gave a very different score.

### 2. Validation Repair

The dataset was analyzed for:

* exact duplicates;
* repeated content;
* class imbalance;
* reciter distribution;
* audio duration;
* anomalous/static recordings;
* train/test acoustic differences.

A fixed 5-fold validation assignment was created and reused for all later serious experiments.

Exact duplicate waveforms were prevented from leaking across folds, and duplicate training examples were represented only once during training.

The important lesson was that **cross-validation was more useful for ranking models than for predicting the absolute leaderboard score**.

The final internal ranking:

```text
ConvNeXt + Delta Features
    >
EfficientNet-B0 + BiGRU/Attention
    >
EfficientNetV2-S
    >
Earlier EfficientNet-B0 variants
```

later matched the private leaderboard ordering.

### 3. Architecture and Representation Experiments

Three major directions were tested in parallel:

#### EfficientNetV2-S

A higher-capacity ImageNet model using the same general log-mel representation.

OOF Macro-F1:

```text
0.9199
```

#### EfficientNet-B0 + BiGRU + Attention

A temporal hybrid that preserved the spectrogram time axis and modeled it explicitly with a bidirectional GRU and temporal attention.

OOF Macro-F1:

```text
0.9302
```

Private Macro-F1:

```text
0.9589
```

#### ConvNeXt-Tiny + Log-Mel + Delta + Delta-Delta

The final winning internal direction.

OOF Macro-F1:

```text
0.9438
```

Private Macro-F1:

```text
0.9608
```

This indicated that **representation design and inductive bias mattered more than simply increasing backbone size**.

## Audio Preprocessing

The final pipeline used:

* sample rate: **16 kHz**
* crop duration: **4 seconds**
* FFT size: **1024**
* hop length: **160**
* mel bins: **128**
* frequency range: **20–8000 Hz**
* delta window: **9**

Longer clips used a conservative activity-aware centering method. If the detector was uncertain, preprocessing safely fell back to the raw midpoint.

Short clips were symmetrically padded.

No training example was removed solely because the audio looked unusual.

## Augmentation

Training used task-preserving audio augmentation:

* center jitter: **±0.20 s**
* gain: **±6 dB**
* mild time-axis warp: **0.96–1.04×**, `p = 0.30`
* up to two frequency masks
* up to two time masks
* synchronized SpecAugment masks across all three spectrogram channels

The time warp was applied before computing delta and delta-delta features so the three channels remained internally consistent.

Validation and test preprocessing were deterministic.

## Training

The final model used:

* **ConvNeXt-Tiny**
* supervised ImageNet-1K initialization
* 5-fold cross-validation
* 14 epochs per fold
* AdamW
* learning rate: `2e-4`
* weight decay: `0.05`
* OneCycle learning-rate schedule
* label smoothing: `0.05`
* dropout: `0.30`
* stochastic depth / drop path: `0.10`
* automatic mixed precision

The best checkpoint for every fold was selected using validation Macro-F1.

On a single NVIDIA T4, the five-fold training stage took roughly **10 minutes**, excluding environment setup and preprocessing.

## Validation Results

Final pooled OOF:

```text
Macro-F1: 0.9438
Accuracy: 0.9413
```

Per-class F1:

| Class  |     F1 |
| ------ | -----: |
| Idgham | 0.9361 |
| Ikhfa  | 0.9442 |
| Iqlab  | 0.9548 |
| Izhar  | 0.9398 |

Fold Macro-F1:

```text
0.9476
0.9263
0.9533
0.9321
0.9600
```

The final model substantially reduced the Idgham/Ikhfa confusion observed in earlier experiments.

## Public vs Private Leaderboard

One of the most useful lessons from the competition was not to overfit to a small public leaderboard.

The final model scored:

```text
Public:  0.5371
Private: 0.9608
```

Several models that looked relatively weak publicly performed extremely well on the much larger private evaluation.

![Submission results](images/submissions.png)

This reinforced the importance of building a consistent validation protocol and using it to compare experiments rather than repeatedly optimizing against public leaderboard noise.

## What I Learned

The main lessons from this competition were:

* **Representation can matter more than backbone size.**
* Temporal derivatives can provide useful information to an image-based audio classifier.
* Strong, label-preserving augmentation should be treated as part of the model design.
* Validation should reflect the structure of the data and should primarily be trusted for comparing experiments.
* Small public leaderboards can be highly misleading.
* Data inspection, preprocessing, normalization, augmentation, and validation can create larger gains than simply scaling the neural network.
* Model diversity is useful only when models make meaningfully different errors; ensembling is not automatically better.

## Repository Contents

```text
.
├── images/
│   ├── leaderboard.png
│   └── submissions.png
├── challenge4_convnext_tiny_deltas.ipynb
└── README.md
```

The notebook contains the complete end-to-end pipeline from competition download and preprocessing through five-fold training, inference, evaluation, artifact generation, and Kaggle submission creation.

## Notes

This repository is intended as a reproducible record of the modelling approach and competition methodology.

The original competition rules restricted the solution to the provided competition data and permitted generic pretrained weights such as ImageNet. The final solution was developed within those constraints.
