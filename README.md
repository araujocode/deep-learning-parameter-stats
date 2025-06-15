# Deep Neural Network Training Parameter Analysis

## Overview

This project explores the monitoring of parameter (weights) and gradient statistics during the training of deep neural networks (DNNs). The primary goal is to provide tools and a methodology for diagnosing the health and stability of the training process by observing how these internal metrics evolve over epochs.

The core idea is that tracking the minimum and maximum values of network parameters and their corresponding gradients can help identify potential issues such as vanishing/exploding gradients or unstable weight updates.

## Key Features

* **Parameter Statistics Extraction:** Implements Python functions using PyTorch to:
  * Collect per-parameter statistics (min/max values of parameter tensors and their gradients).
  * Aggregate these into global minimum and maximum values across all parameters and all gradients for each training epoch.
* **Training Integration:** Demonstrates how to integrate this statistics collection mécanisme into a standard PyTorch training loop.
* **Visualization:** Plots the evolution of:
  * Training Loss
  * Training Accuracy
  * Global minimum/maximum parameter values
  * Global minimum/maximum gradient values
* **Example Implementation:** Uses a ResNet-18 model trained on dummy data to showcase the functionality.

## How It Works

1. **Model and Data:** A ResNet-18 model is initialized (either with or without pre-trained weights, for the training loop example, it's typically trained from scratch or on a specific task). Dummy data is generated for demonstration purposes.
2. **Statistics Functions:**
    * `get_param_stats(model)`: Iterates through all named parameters of the input PyTorch model. For each parameter, it records its minimum and maximum data values. If gradients have been computed (i.e., after `loss.backward()`), it also records the minimum and maximum gradient values.
    * `get_global_stats(param_stats_dict)`: Takes the dictionary produced by `get_param_stats` and finds the overall minimum and maximum values across all parameters, and similarly for all gradients.
3. **Training Loop:**
    * A standard training loop is set up (epochs, data loader, optimizer, loss function).
    * In each epoch, after the `loss.backward()` call (which computes gradients) and `optimizer.step()` (which updates parameters), the `get_param_stats` and `get_global_stats` functions are called to retrieve the current statistics.
    * These statistics, along with loss and accuracy, are stored.
4. **Live Plotting:** During training (typically in a Jupyter/Colab environment), the collected metrics (loss, accuracy, and the four parameter/gradient global statistics) are plotted, updating after each epoch to provide a live view of the training dynamics.

## Purpose

The main purpose is to provide a practical tool for developers and researchers to:

* Gain deeper insights into the training dynamics of their neural networks.
* Detect early signs of training instability (e.g., exploding or vanishing gradients).
* Make more informed decisions about hyperparameter tuning, model architecture adjustments, or debugging training issues.

## Running the Code

The primary implementation is provided as a Google Colaboratory notebook (`.ipynb` file).

1. Open the notebook in Google Colab.
2. Ensure the necessary Python libraries are available (PyTorch, Matplotlib). These are typically pre-installed in Colab environments or can be installed via `pip`.
3. Run the cells sequentially to define the functions, set up the model and data, and execute the training loop with live plotting.

*The notebook can be found [here](https://colab.research.google.com/drive/1xI63qLtd8CjpIuhiIMxgFmy3RiOn7RJ3?usp=drive_link).)*
