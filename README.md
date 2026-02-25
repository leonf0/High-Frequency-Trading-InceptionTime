# High Frequency Trading - InceptionTime (Convolutional Neural Networks)

A 3-class classification task that uses a variant of the **"InceptionTime"** Convolutional Neural Network architecture to predict short-term directional movements in mid price based on raw-LOB data.

# Project Overview

In this project we adapt the **"InceptionTime"** CNN architecture and apply it to the task of directional classification in a high frequency trading environment, using the FI-2010 dataset. The **"InceptionTime"** architecture was proposed in the 2019 paper “InceptionTime: Finding AlexNet for Time Series Classification.”. The architecture applies convolutions at several receptive field sizes simultaneously to look for short, medium, and long term patterns in a time series. Unlike many traditional time-series methods this allows it to identify multiple temporal structures that a single-scale window would miss.

The main architectural changes we make are adding dropout layers after every inception block and at the head, which helps regularize and prevent overfitting considering the noisiness of LOB data and the relatively small dataset (362,401 rows of training data across 9 days), and reducing the number of inception blocks from 5 to 3 which improves the models ability to generalise given the dataset size and the signal to noise ratio of LOB data, consistent with the generalisation literature on deep networks in small-data regimes. This architectural change also allows training to run significantly faster.

# Model Architecture

<p align="center">
  <img src="assets/inceptiontime_lob_architecture.png" alt="" width="80%"/>
</p>

## Input
We have out input as a 3D tensor, it is a batch of LOB snapshots (we use batches of 64 for training and 128 for evaluation) where 40 is the feature dimension (10 price/volume levels × bid & ask × 2) and 100 is the temporal sequence length. The temporal axis is what the convolutions slide over.

## ResidualBlock (×2)

We use a structure  that wraps three InceptionBlocks inside a residual skip connection. The shortcut is a 1D CNN with kernel size 1 and a batch normalisation layer, this projects the input to match the output channel count. We then add them elementwise to the output of the third InceptionBlock before a final ReLU activation. This stabilises gradients and lets the model learn identity mappings if deeper processing isn't useful.

## InceptionBlock

#### Bottleneck 

We use a 1D CNN with a kernel size of 1 that compresses the channel dimension before we move to the expensive multi-scale convolutions. Borrowed from Inception-v2 / ResNet bottleneck design — reduces compute cost while mixing cross-channel information. Skipped only if in_channels == 1 (never triggered here).

3. Three parallel Conv1d branches (k=3, k=7, k=11)
Each branch processes the bottleneck output with a different kernel size, capturing temporal dependencies at different scales. k=3 picks up very local micro-structure dynamics; k=7 and k=11 capture progressively longer-range patterns (order book evolution over more timesteps). Padding is set to k//2 to preserve the temporal dimension.
4. AvgPool1d + Conv1d branch (the "pooling path")
An AvgPool1d(k=3, stride=1) followed by a Conv1d(k=1) applied directly to the pre-bottleneck input (not the bottleneck output). This is the standard Inception pooling branch — it captures smoothed, local temporal aggregations and acts as a mild regulariser.
5. Concatenate → 4 × 32 = 128 channels
The four branch outputs are concatenated along the channel axis. The resulting tensor has 128 channels and the same temporal length (sequence dimension is preserved throughout).
6. BatchNorm1d → ReLU → Dropout(p=0.25)
Standard post-activation normalisation. Dropout at 0.25 regularises within each block.

Global Average Pooling — (batch, 128, 100) → (batch, 128)
AdaptiveAvgPool1d(1) averages across the entire temporal axis, collapsing the 100-timestep dimension to a single vector per sample. This is a key design choice: the model learns a temporal summary of the LOB dynamics rather than just the final snapshot.

Classification Head

Dropout(p=0.35) — slightly heavier dropout before the final linear layer for additional regularisation.
Linear(128 → 3) — projects to three logits corresponding to the three LOB mid-price movement classes: Down / Stationary / Up.


## Results Summary

<p align="center">
  <img src="assets/classification_report.png" alt="" width="80%"/>
</p>

<p align="center">
  <img src="assets/confusion_matrix_InceptionTime.png" alt="" width="80%"/>
</p>
