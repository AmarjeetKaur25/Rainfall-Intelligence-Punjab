
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Rainfall_Data_LL.csv"
ANNUAL_MODEL_PATH = BASE_DIR / "models" / "rainfall_model.pkl"
MONTHLY_MODEL_PATH = BASE_DIR / "models" / "monthly_rainfall_model.pkl"

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

annual_model = joblib.load(ANNUAL_MODEL_PATH)
monthly_model = joblib.load(MONTHLY_MODEL_PATH)

# --------------------------------------------------
# LOAD AND CLEAN DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

subdivision_column = "SUBDIVISION"

df = df[
    df[subdivision_column].astype(str).str.contains(
        "Punjab", case=False, na=False
    )
].copy()

MONTHS = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
]

MONTH_NAMES = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")

for month in MONTHS:
    df[month] = pd.to_numeric(df[month], errors="coerce")

df = df.dropna(subset=["YEAR"])
df["YEAR"] = df["YEAR"].astype(int)
df = df.sort_values("YEAR")

df["ANNUAL_RAINFALL"] = df[MONTHS].sum(axis=1)

# --------------------------------------------------
# HISTORICAL STATISTICS
# --------------------------------------------------

average_rainfall = float(df["ANNUAL_RAINFALL"].mean())
rainfall_std = float(df["ANNUAL_RAINFALL"].std())

# Classification thresholds: 20% deviation
lower_limit = average_rainfall * 0.80
upper_limit = average_rainfall * 1.20

highest_rainfall = float(df["ANNUAL_RAINFALL"].max())
lowest_rainfall = float(df["ANNUAL_RAINFALL"].min())

highest_year = int(
    df.loc[df["ANNUAL_RAINFALL"].idxmax(), "YEAR"]
)

lowest_year = int(
    df.loc[df["ANNUAL_RAINFALL"].idxmin(), "YEAR"]
)

dataset_start_year = int(df["YEAR"].min())
dataset_end_year = int(df["YEAR"].max())

# --------------------------------------------------
# MONTHLY HISTORICAL STATISTICS
# --------------------------------------------------

monthly_averages = df[MONTHS].mean().to_dict()
monthly_std = df[MONTHS].std().to_dict()

monthly_average_values = [
    round(float(monthly_averages[m]), 2)
    for m in MONTHS
]

# --------------------------------------------------
# DATE
# --------------------------------------------------

current_year = datetime.now().year
current_month = datetime.now().month

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",

        average_rainfall=round(average_rainfall, 2),

        highest_rainfall=round(highest_rainfall, 2),
        lowest_rainfall=round(lowest_rainfall, 2),

        highest_year=highest_year,
        lowest_year=lowest_year,

        years=df["YEAR"].tolist(),

        monthly_averages=monthly_average_values,

        dataset_start_year=dataset_start_year,
        dataset_end_year=dataset_end_year,

        current_year=current_year,
        current_month=current_month
    )

# --------------------------------------------------
# ANNUAL CATEGORY
# --------------------------------------------------

def classify_annual_rainfall(prediction):

    if prediction < lower_limit:
        return "Below Normal"

    elif prediction > upper_limit:
        return "Above Normal"

    else:
        return "Normal"

# --------------------------------------------------
# MONTHLY CATEGORY
# --------------------------------------------------

def classify_monthly_rainfall(prediction, month):

    historical_mean = float(monthly_averages[month])

    # 20% deviation from historical monthly average
    monthly_lower_limit = historical_mean * 0.80
    monthly_upper_limit = historical_mean * 1.20

    if prediction < monthly_lower_limit:
        category = "Below Normal"

    elif prediction > monthly_upper_limit:
        category = "Above Normal"

    else:
        category = "Normal"

    return category, monthly_lower_limit, monthly_upper_limit

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        prediction_type = data.get("type")

        # ==========================================
        # ANNUAL PREDICTION
        # ==========================================

        if prediction_type == "annual":

            year = int(data.get("year"))

            if year not in [current_year + 1, current_year + 2]:
                return jsonify({
                    "error": "Please select a valid future year."
                }), 400

            input_data = pd.DataFrame(
                [[monthly_averages[m] for m in MONTHS]],
                columns=MONTHS
            )

            predicted_rainfall = float(
                annual_model.predict(input_data)[0]
            )

            category = classify_annual_rainfall(
                predicted_rainfall
            )

            return jsonify({

                "success": True,

                "type": "annual",

                "year": year,

                "rainfall": round(predicted_rainfall, 2),

                "category": category,

                "historical_average": round(
                    average_rainfall, 2
                ),

                "lower_limit": round(lower_limit, 2),

                "upper_limit": round(upper_limit, 2),

                "disclaimer": (
                    "Experimental annual rainfall estimate "
                    "based on historical patterns. "
                    "Not an official weather forecast."
                )

            })

        # ==========================================
        # MONTHLY PREDICTION
        # ==========================================

        elif prediction_type == "monthly":

            year = int(data.get("year"))
            month = int(data.get("month"))

            # Validate year
            if year not in [current_year, current_year + 1]:

                return jsonify({
                    "error": "Please select a valid prediction year."
                }), 400

            # Validate month
            if month < 1 or month > 12:

                return jsonify({
                    "error": "Invalid month selected."
                }), 400

            # Prevent predicting completed months
            if year == current_year and month <= current_month:

                return jsonify({
                    "error": (
                        "Please select a future month "
                        "for the current year."
                    )
                }), 400

            selected_month = MONTHS[month - 1]
            selected_month_name = MONTH_NAMES[month - 1]

            # --------------------------------------
            # MONTHLY MODEL INPUT
            # --------------------------------------

            previous_month = MONTHS[(month - 2) % 12]

            previous_month_rainfall = float(
                monthly_averages[previous_month]
            )

            previous_year_rainfall = float(
                monthly_averages[selected_month]
            )

            input_data = pd.DataFrame([{

                "MONTH_NUM": month,

                "PREVIOUS_MONTH_RAINFALL":
                    previous_month_rainfall,

                "PREVIOUS_YEAR_RAINFALL":
                    previous_year_rainfall

            }])

            predicted_rainfall = float(
                monthly_model.predict(input_data)[0]
            )

            # --------------------------------------
            # MONTH-SPECIFIC CLASSIFICATION
            # --------------------------------------

            category, monthly_lower_limit, monthly_upper_limit = (
                classify_monthly_rainfall(
                    predicted_rainfall,
                    selected_month
                )
            )

            historical_monthly_average = float(
                monthly_averages[selected_month]
            )

            historical_monthly_std = float(
                monthly_std[selected_month]
            )

            return jsonify({

                "success": True,

                "type": "monthly",

                "year": year,

                "month": month,

                "month_name": selected_month_name,

                "rainfall": round(
                    predicted_rainfall, 2
                ),

                "category": category,

                "historical_average": round(
                    historical_monthly_average, 2
                ),

                "historical_monthly_average": round(
                    historical_monthly_average, 2
                ),

                "historical_monthly_std": round(
                    historical_monthly_std, 2
                ),

                "lower_limit": round(
                    monthly_lower_limit, 2
                ),

                "upper_limit": round(
                    monthly_upper_limit, 2
                ),

                "disclaimer": (
                    "Experimental monthly rainfall estimate "
                    "using historical patterns and statistical "
                    "thresholds. Not an official weather forecast."
                )

            })

        # ==========================================
        # INVALID REQUEST
        # ==========================================

        else:

            return jsonify({
                "error": "Invalid prediction type."
            }), 400

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# --------------------------------------------------
# RUN APP
# --------------------------------------------------

if __name__ == "__main__":

    app.run(debug=True)
