# Forest Fire Prediction and Simulation using AI/ML – ISRO BAH 2025

## Problem Statement
This project addresses **Problem Statement-1** of ISRO BAH 2025:  
**Simulation/Modelling of Forest Fire Spread using AI/ML Techniques**

**Objectives:**
1. Predict forest fire-prone areas for the next day (binary classification: fire/no fire) for a region like Uttarakhand.
2. Simulate fire spread from high-risk areas over short-term intervals: 1h, 2h, 3h, 6h, and 12h.

All outputs are georeferenced raster maps at 30m resolution.

## Approach Summary

### Objective 1: Fire Risk Prediction (Next-Day)
- Model: Lightweight UNet with skip connections
- Input: 13-band raster stack (NDVI, rainfall, temperature, humidity, LULC, slope, etc.)
- Training: 128×128 patches, Combo Loss (BCE + Soft IoU)
- Post-Processing: Binary thresholding and noise removal
- Output: Raster map showing next-day high-risk fire zones

### Objective 2: Fire Spread Simulation
- Model: Hybrid Sequence-aware LSTM + Spatial CNN
- Input:
  - 5-time step sequence of VIIRS fire masks and ERA5 wind data
  - 5×5 vegetation and terrain context (NDVI, NBR, slope)
  - 15-class directional embedding (wind-slope alignment)
- Post-Processing: Physics-informed Rate of Spread (ROS) corrections
- Simulation: Probabilistic Cellular Automata (CA)
- Output: Time-stepped rasters (1h to 12h) and optional animation

---

## Datasets Used

| Dataset            | Features                        | Source                |
|--------------------|----------------------------------|------------------------|
| ERA5               | Wind speed and direction         | Copernicus             |
| VIIRS Fire Mask    | Historical fire detection        | NASA FIRMS             |
| DEM                | Slope, Aspect                    | Bhoonidhi              |
| NDVI, NBR          | Vegetation and burn index        | Sentinel-2 (GEE)       |
| LULC               | Fuel classification              | Bhuvan (NRSC)          |

All rasters are aligned and resampled to 30m resolution.

---

## Libraries and Tools
- PyTorch, torchvision
- rasterio, gdal, opencv, matplotlib, scipy, imageio
- numpy, scikit-learn
- Compatible with CUDA and Google Colab

---

## Evaluation Metrics

| Metric              | Purpose                                      |
|---------------------|----------------------------------------------|
| IoU, Precision, Recall | Fire prediction accuracy                   |
| Kappa Score         | Fire spread agreement with VIIRS             |
| Hausdorff Distance  | Spatial deviation of spread simulation       |
| ROC-AUC (optional)  | Probabilistic burn map evaluation            |

---

## Unique Contributions
- Physics-informed post-processing for realistic fire dynamics
- Hybrid AI model combining temporal and spatial inputs
- Probabilistic Cellular Automata for dynamic fire spread
- Modular, scalable pipeline with real-world application potential

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/isro-forest-fire-simulation.git
cd isro-forest-fire-simulation
```

## 2. Prepare Data
Place all raster files in the data/ folder.
Ensure all rasters are resampled to 30m and correctly georeferenced.

## 3. Run the Pipeline
Start with ISRO_EDA_&_Data_Processing.ipynb to prepare and patchify inputs.
Train the UNet model using ISRO_Model_F.ipynb for fire zone prediction.
Use ISRO_CA.ipynb to run fire spread simulation over multiple time intervals.

## Sample Outputs
Fire probability map for next-day risk zones
Simulated spread maps: fire_spread_1h.tif, fire_spread_2h.tif, ..., fire_spread_12h.tif
Optional animation: fire_spread_animation.gif
