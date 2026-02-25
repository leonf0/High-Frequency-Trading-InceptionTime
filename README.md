# High Frequency Trading - InceptionTime (Convolutional Neural Networks)

A 3-class classification task that uses a variant of the **"InceptionTime"** Convolutional Neural Network architecture to predict short-term directional movements in mid price based on raw-LOB data.

## Project Overview

In this project we adapt the **"InceptionTime"** CNN architecture and apply it to the task of directional classification in a high frequency trading environment, using the FI-2010 dataset. The **"InceptionTime"** architecture was proposed in the 2019 paper “InceptionTime: Finding AlexNet for Time Series Classification.”. The architecture applies convolutions at several receptive field sizes simultaneously to look for short, medium, and long term patterns in a time series. Unlike many traditional time-series methods this allows it to identify multiple temporal structures that a single-scale window would miss.

The main architectural changes we make are adding dropout layers after every inception block and at the head, which helps regularize and prevent overfitting considering the noisiness of LOB data and the relatively small dataset (362,401 rows of training data across 9 days), and reducing the number of inception blocks from 5 to 3 which improves the models ability to generalise given the dataset size and the signal to noise ratio of LOB data, consistent with the generalisation literature on deep networks in small-data regimes. This architectural change also allows training to run significantly faster.

## Model Architecture

## Results Summary
