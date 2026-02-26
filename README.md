# High Frequency Trading - InceptionTime (Convolutional Neural Networks)

A 3-class classification task that uses a variant of the **"InceptionTime"** Convolutional Neural Network architecture to predict short-term directional movements in mid price based on raw-LOB data.

# Project Overview

In this project we adapt the **"InceptionTime"** CNN architecture and apply it to the task of directional classification in a high frequency trading environment, using the FI-2010 dataset where we have a 10 level LOB snapshot as the input features and a target label of the directional movement after 10 events. 

The **"InceptionTime"** architecture was proposed in the 2019 paper “InceptionTime: Finding AlexNet for Time Series Classification.”. The architecture applies convolutions at several receptive field sizes simultaneously to look for short, medium, and long term patterns in a time series. Unlike many traditional time-series methods this allows it to identify multiple temporal structures that a single-scale window would miss.

The main architectural changes we make are adding dropout layers after every inception block and at the head, which helps regularize and prevent overfitting considering the noisiness of LOB data and the relatively small dataset (362,401 rows of training data across 9 days). In the inception blocks we use a dropout of 0.25, and on the classification head we use a dropout of 0.35. 

We use an initial learning rate of 0.001, with a scheluder that halves the learning rate that halves the learning rate after 3 training epochs without improvement. Additionally we use 3 kernels of sizes 3, 7 and 11. All these hyperparameter choices were determined using grid-search on a the validation dataset we split from the training set prior to training. The model has 221,059 total trainable paramaters which are optimized using the Adaptive Moment Estimater (ADAM) optimizer.



# Model Architecture

<p align="center">
  <img src="assets/inceptiontime_lob_architecture.png" alt="" width="80%"/>
</p>

## Input
We have out input as a 3D tensor, it is a batch of LOB snapshots (we use batches of 64 for training and 128 for evaluation) where 40 is the feature dimension (10 price/volume levels × bid & ask × 2) and 100 is the temporal sequence length. The temporal axis is what the convolutions slide over.

## ResidualBlock (×2)

We use a structure  that wraps three InceptionBlocks inside a residual skip connection. The shortcut is a 1D CNN with kernel size 1 and a batch normalisation layer, this projects the input to match the output channel count. We then add them elementwise to the output of the third InceptionBlock before a final ReLU activation. This stabilises gradients and lets the model learn identity mappings if deeper processing isn't useful.

## InceptionBlock

#### Bottleneck 

We use a 1D CNN with a kernel size of 1 that compresses the channel dimension before we move to the expensive multi-scale convolutions. This reduces compute cost and mixes the cross-channel information.

#### Three parallel Conv1d branches

We use 3 parallel 1-D CNN branches to process the bottleneck output each branch with a different kernel size, to capture temporal dependencies at different scales. The first kernel size we use is 3 to pick up on local micro-structure patterns, and then the other branches use kernels of sizes 7 and 11 to pick up progressively longer patterns. We set padding to k//2 where k is the kernel size.

#### Average Pooling and 1D-CNN branch 

We use an average pooling layer with kernel size 3 and stride length 1, followed by a 1D CNN (with kernel size 1) which is applied directly to the pre-bottleneck input. This captures smoothed, local temporal aggregations and can act as a mild regulariser.

#### Concatenate
We then concatenate the outputs of the four branch along the channel axis. The resulting tensor has 128 channels and the same temporal length (100).

#### Batch Normalisation → ReLU → Dropout(p=0.25)
We then use batch normalisation followed by ReLU activation and set dropout at 0.25 in order to regularise and force the model to learn high-signal features. This is applied within each block.

## Global Average Pooling
We then averages across the entire temporal axis, which collapses the 100-timestep dimension into a single vector per sample so that the model learns a temporal summary of the LOB dynamics.

## Classification Head
We set a slightly heavier dropout (p=0.35) before the final linear layer, which projects the data to three logits corresponding to the three LOB mid-price movement classes (Down / Stationary / Up).

## Results Summary

<p align="center">
  <img src="assets/classification_report.png" alt="" width="80%"/>
</p>

The above table shows the models performance, evaluated on a test set of 1 day of data. The model shows strong predictive strength achieving a macro-F1 score of 0.7013, which outperforms a baseline Ridge Regression model trained and tested on the same datatset (0.44 macro-F1).

<p align="center">
  <img src="assets/confusion_matrix_InceptionTime.png" alt="" width="80%"/>
</p>

The above figure demonstates the confusion matrix of the models predictions compares to the actual classes. One particular trait of the model that this shows is the fact that the error structure is economically sensible. The majority of errors are confusion between movement and stationary classification, not directional flips:

Down misclassified as Stationary: 2985
Up misclassified as Stationary: 2559

Whereas true directional confusion is comparatively rare:

Down misclassified as Up: 1270
Up misclassified as Down: 1248

This is very important in a trading context, predicting "stationary" when the price moves is a simply missed opportunity on the other hand if the model predicts "Down" when the price goes "Up" it would result in a loss. Such a error profile on the model reinforces the models predictive strength and viablity in a high-frequency trading context, and indicates that a potential threshold filter on the stationary class confidence could potentially be applied to improve signal precision.
