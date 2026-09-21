
# Self-Supervised Speech Representations and Residual TCNs for Subspace-Compressed Articulatory Inversion

This repository contains the official PyTorch implementation of the paper "Self-Supervised Speech Representations and Residual TCNs for Subspace-Compressed Articulatory Inversion".

## Project Overview

Acoustic-to-Articulatory (A2A) inversion aims to reconstruct the continuous physical movements of the vocal tract (tongue, lips, jaw, velum) directly from a speech audio signal. Traditional approaches struggle with three primary issues: the high dimensionality and noise of Electromagnetic Articulography (EMA) data, acoustic-articulatory non-linearity, and the "one-to-many" mapping problem where different vocal tract configurations can produce acoustically similar sounds.

This project solves these challenges by combining robust Self-Supervised Speech Representations (via WavLM) with a customized Residual Temporal Convolutional Network (TCN). Furthermore, it demonstrates a critical finding: attempting to predict all 12 EMA channels simultaneously often degrades overall model performance due to the inclusion of low-variance or inactive articulators ("dead channels"). We introduce a pipeline that dynamically compresses the output into a variance-optimized 5D articulatory subspace, yielding highly accurate, low-latency tracking across three standard benchmark datasets (MOCHA, USC-TIMIT, and MNGU0).

## Architecture & Methodology

### 1. Acoustic Feature Extraction (WavLM)
Instead of using traditional spectrograms or Mel-Frequency Cepstral Coefficients (MFCCs), we extract high-level embeddings from the last hidden state of WavLM-Large. WavLM is a self-supervised transformer trained on thousands of hours of speech, providing highly contextualized, noise-robust phonetic representations at roughly 50 frames per second. This shifts the heavy lifting of acoustic modeling from the inversion network to the pre-trained transformer.

### 2. The Residual TCN (A2A_TCN_v2)
The core inversion model is a 1D Dilated Temporal Convolutional Network carefully tuned for kinematic sequence mapping.
* **Dilations:** By scaling dilation rates (1, 2, 4, 8, 16, 32), the network exponentially expands its receptive field. This allows it to capture long-term phonetic dependencies (coarticulation) without losing frame-by-frame temporal resolution.
* **Residual Blocks:** Each block utilizes depth-wise grouped convolutions, GroupNorm (which outperforms BatchNorm on small batch sequence data), and GELU activations. Residual skip-connections ensure smooth gradient flow during training.

### 3. Subspace Optimization
EMA datasets track various coils, but not all coils move significantly during normal speech (e.g., reference coils or off-axis dimensions). The pipeline evaluates raw coordinate variance and baseline prediction correlations (PCC) to dynamically filter out "dead" channels. Projecting the modeling task from 12D to the optimal 5D subspace prevents the network from wasting capacity on static noise.

### 4. Multi-Objective Kinematic Loss
Tracking physical movement requires more than just minimizing coordinate distances. We use a custom `ArticulatoryLoss` combining four distinct objectives:
1. **L1 Loss (1.0):** Standard mean absolute error for frame-by-frame coordinate accuracy, proving more robust to outliers than MSE.
2. **Variance Loss (0.1):** Penalizes the model if the predicted trajectory is overly smoothed and fails to match the dynamic range (variance) of the ground truth.
3. **Velocity Loss (0.2):** Computes the first derivative (frame-to-frame difference) to ensure the speed and smoothness of the predicted articulators match human biomechanics.
4. **CCC Loss (0.5):** Uses the Concordance Correlation Coefficient to maximize the morphological shape agreement between the predicted and true trajectories.

## Project Structure
```
a2a-inversion/
├── requirements.txt           # Python dependencies
├── utils/
│   ├── __init__.py
│   ├── signal_processing.py   # Resamples audio to 16kHz and EMA to 50Hz (WavLM sync)
│   ├── post_processing.py     # Savitzky-Golay smoothing and validation-based lag-correction
│   └── metrics.py             # Calculates Pearson (PCC) and Concordance (CCC) metrics
├── data/
│   ├── __init__.py
│   ├── extraction.py          # HuggingFace WavLM integration and GPU batching
│   └── dataset.py             # Dataloaders with sequence padding and standard scaling
├── models/
│   ├── __init__.py
│   ├── architectures.py       # A2A_TCN_v2 Residual Dilated TCN definition
│   └── loss.py                # Multi-Objective Kinematic Loss function
├── scripts/
│   ├── __init__.py
│   ├── evaluate.py            # Inference un-padding and metric aggregation
│   ├── train.py               # Early-stopping and plateau-based learning rate scheduling
│   └── plotting.py            # Hexbin densities, CDFs, and trajectory visualization
└── main.py                    # Master execution orchestrator for end-to-end runs
```
Installation
Clone the repository:

```Bash
git clone [https://github.com/yourusername/Self-Supervised-Speech-Representations-and-Residual-TCNs-for-Subspace-Compressed-Articulatory-Inv.git](https://github.com/yourusername/Self-Supervised-Speech-Representations-and-Residual-TCNs-for-Subspace-Compressed-Articulatory-Inv.git)
cd Self-Supervised-Speech-Representations-and-Residual-TCNs-for-Subspace-Compressed-Articulatory-Inv
```
Create a virtual environment and install dependencies:

``` Bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```
Note: Feature extraction relies on HuggingFace's WavLM-Large. A CUDA-enabled GPU with at least 8GB VRAM is highly recommended.

Usage
The entire pipeline—from raw data ingestion, feature extraction, 12D baseline training, Subspace Optimization, 5D retraining, and plotting—is orchestrated through a single entry point.

Place your .npz dataset files (e.g., mocha_baseline_12D.npz) anywhere in the working directory, and run:

```Bash
python main.py
```
What happens under the hood?
Dynamic Resampling: Raw audio is resampled to 16kHz for WavLM. High-speed EMA data (e.g., 500Hz for MOCHA) is carefully downsampled to 50Hz to align perfectly with the transformer's acoustic output frames.

12D Baseline: The network is initially trained to predict all 12 EMA channels.

Subspace Optimization: The script analyzes the EMA standard deviations and validation PCCs from the 12D model to dynamically identify the top 5 most active, structurally reliable articulators.

5D Subset Training: A new model is trained exclusively on this optimal subset, significantly reducing RMSE by ignoring noise from dead channels.

Lag Correction: A cross-correlation pass on the validation set corrects minor millisecond delays in the predictions prior to final metric evaluation.

Outputs Generated
All visual analytics are exported to the paper_results/ directory during script execution:

Zoomed Trajectories: Millisecond-level side-by-side tracking of Ground Truth vs. Full 12D Baseline vs. Optimal 5D prediction.

Multi-Channel Facets: A stacked overview of all predicted optimal channels.

Error CDFs: Cumulative distribution functions showing the percentage of frames falling under absolute millimeter error thresholds.

Scatter Density: Hexbin correlation plots analyzing the prediction distribution density against the ideal theoretical regression line.

Correlation Heatmaps: Quantifies the inter-articulator synergies (e.g., jaw vs. lower lip) captured by the model.
