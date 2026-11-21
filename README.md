
# 🌧️ Rainfall Intelligence – Punjab

An experimental machine-learning-based web application for historical rainfall analysis and annual/monthly rainfall estimation in Punjab.

## 📌 About the Project

Rainfall Intelligence – Punjab is a data-driven web application developed to explore historical rainfall patterns in Punjab and generate experimental rainfall estimates using Machine Learning.

The platform combines historical data analysis, Random Forest regression models and an interactive web dashboard to display rainfall estimates, historical averages and rainfall categories.

## 🎯 Objectives

- Analyze historical rainfall data for Punjab.
- Visualize historical rainfall statistics.
- Generate experimental annual rainfall estimates.
- Generate experimental month-specific rainfall estimates.
- Classify rainfall estimates as Below Normal, Normal or Above Normal.
- Provide an interactive and user-friendly dashboard.

## 🚀 Features

- Historical rainfall dashboard.
- Annual rainfall estimation.
- Monthly rainfall estimation.
- Historical average rainfall comparison.
- Rainfall classification.
- Statistical threshold display.
- Interactive web interface.
- Experimental ML-based estimation.

## 🧠 Machine Learning Models

### Annual Rainfall Model
- Algorithm: Random Forest Regressor
- Input: 12 monthly rainfall features
- Output: Experimental annual rainfall estimate

### Monthly Rainfall Model
- Algorithm: Random Forest Regressor
- Features:
  - MONTH_NUM
  - PREVIOUS_MONTH_RAINFALL
  - PREVIOUS_YEAR_RAINFALL
- Output: Experimental monthly rainfall estimate

## 📊 Dataset

The project uses historical rainfall data from `Rainfall_Data_LL.csv`, filtered to records associated with the Punjab subdivision.

- Historical period: 1901–2017
- Data processing: Pandas
- Annual rainfall: Calculated from the twelve monthly rainfall values

## 🛠️ Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- Joblib
- HTML
- CSS
- JavaScript

## 📁 Project Structure

```text
Rainfall_Crop_Risk_Punjab/
├── data/
│   └── Rainfall_Data_LL.csv
├── models/
│   ├── rainfall_model.pkl
│   └── monthly_rainfall_model.pkl
├── notebooks/
│   └── 01_data_inspection.ipynb
├── static/
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AmarjeetKaur25/Rainfall-Intelligence-Punjab.git
```

### 2. Navigate to the Project Directory

```bash
cd Rainfall-Intelligence-Punjab
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

### 5. Open the Website

Visit:

http://127.0.0.1:5000

## 📈 Model Evaluation

The annual model was evaluated using:

- Mean Absolute Error (MAE): 125.50 mm
- Mean Squared Error (MSE): 21,980.64
- Root Mean Squared Error (RMSE): Approximately 148.26 mm
- R² Score: -0.143

The model demonstrated a reduction in MAE compared with the recorded baseline, but its predictive performance remains limited. Further improvement is required.

## ⚠️ Disclaimer

This project is an experimental machine-learning application based on historical rainfall patterns.

The estimates are not official meteorological forecasts. The dataset does not include live weather observations, and the predictions should not be used as the sole basis for agricultural, emergency or water-management decisions.

The rainfall categories and thresholds are project-defined statistical comparisons and are not official meteorological classifications.

## 👥 Team Members

- [Team Member 1 – Name]
- [Team Member 2 – Name]
- [Team Member 3 – Name]
- [Team Member 4 – Name]

## 🔮 Future Scope

- Integration of updated meteorological datasets.
- District-level rainfall analysis.
- Additional climate-related input features.
- Improved machine-learning and time-series models.
- Model validation and uncertainty estimation.
- Integration with trusted weather data sources.

---

Developed as a collaborative academic Machine Learning and Web Development project.
