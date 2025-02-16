# Vehicle Acoustics Data Processing and Modeling

## Overview
This repository contains code for processing acoustics data from a vehicle using signal processing techniques and machine learning models. The project involves three main tasks:

1. **Processing acoustics data** from a vehicle using known signal processing techniques.
2. **Recreating maps** of the vehicle's location based on GPS data.
   - Previous maps are stored in the `maps/` folder.
3. **Training machine learning models** to predict engine RPM from acoustics data:
   - A **Fully Connected Neural Network (FCNN)**
   - A **Long Short-Term Memory (LSTM) network**

## Features
- **Signal Processing:** Implements various filters and transforms to extract meaningful insights from acoustic signals.
- **GPS Data Visualization:** Generates maps showing vehicle paths.
- **Machine Learning Models:** Trains deep learning models to predict engine RPM based on processed audio data.

## Installation
Clone the repository and install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
To process the acoustics data, run:
```bash
python processing.py # Make sure to specify the correct data path before running.
```
Train the LSTM with:
```bash
python train_lstm.py
```

Train the FCN with:
```bash
python train_fcn.py
```

## Folder Structure
- **`data/`** – Stores raw and processed acoustics data.  
- **`maps/`** – Contains previously generated vehicle maps.  
- **`plots/`** – Stores training plots from previous runs.  
- **`models/`** – Saved machine learning models.  
- **`utils/`** – Helper functions for processing and visualization.  