import json
import os
import threading
from pathlib import Path

import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from neuralforecast import NeuralForecast


# -----------------------------------------------------
# Paths
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = Path(
    os.getenv(
        "MODEL_DIR",
        BASE_DIR / "model" / "itransformer"
    )
)

DATA_PATH = Path(
    os.getenv(
        "DATA_PATH",
        BASE_DIR / "data" / "latest_168h.csv"
    )
)

FRONTEND_DIR = BASE_DIR / "frontend"

METRICS_PATH = MODEL_DIR / "metrics.json"


# -----------------------------------------------------
# Application
# -----------------------------------------------------

app = FastAPI(
    title="ReviewPulse API",
    description="iTransformer review activity forecasting API",
    version="1.0"
)


# -----------------------------------------------------
# Global application state
# -----------------------------------------------------

forecast_lock = threading.Lock()

nf = None
history_df = None
forecast_df = None
model_column = None

INPUT_SIZE = 168
HORIZON = 24


# -----------------------------------------------------
# Load model
# -----------------------------------------------------

def load_model():

    global nf
    global INPUT_SIZE
    global HORIZON

    print("Loading iTransformer model...")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model folder not found: {MODEL_DIR}"
        )

    # Load checkpoint onto CPU for Docker deployment.
    nf = NeuralForecast.load(
        path=str(MODEL_DIR),
        map_location="cpu"
    )

    # Force CPU inference if the checkpoint was trained on GPU.
    for model in nf.models:

        if hasattr(model, "trainer_kwargs"):

            model.trainer_kwargs["accelerator"] = "cpu"
            model.trainer_kwargs["devices"] = 1
            model.trainer_kwargs["precision"] = "32-true"
            model.trainer_kwargs["logger"] = False
            model.trainer_kwargs["enable_progress_bar"] = False

    first_model = nf.models[0]

    INPUT_SIZE = int(
        getattr(first_model, "input_size", 168)
    )

    HORIZON = int(
        getattr(first_model, "h", 24)
    )

    print("Model loaded.")
    print("Input size:", INPUT_SIZE)
    print("Forecast horizon:", HORIZON)


# -----------------------------------------------------
# Load deployment data
# -----------------------------------------------------

def load_data():

    global history_df

    print("Loading inference data...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    unnamed = [
        c for c in df.columns
        if c.lower().startswith("unnamed")
    ]

    if unnamed:
        df = df.drop(columns=unnamed)

    required = {"unique_id", "ds", "y"}

    if not required.issubset(df.columns):
        raise ValueError(
            "CSV must contain unique_id, ds and y."
        )

    df = df[
        ["unique_id", "ds", "y"]
    ].copy()

    df["ds"] = pd.to_datetime(df["ds"])

    df["y"] = pd.to_numeric(
        df["y"],
        errors="raise"
    )

    df = df.sort_values(
        ["unique_id", "ds"]
    ).reset_index(drop=True)

    counts = df.groupby("unique_id").size()

    if not (counts == INPUT_SIZE).all():

        raise ValueError(
            f"Every series must contain exactly "
            f"{INPUT_SIZE} observations."
        )

    expected_series = getattr(
        nf.models[0],
        "n_series",
        None
    )

    actual_series = df["unique_id"].nunique()

    if (
        expected_series is not None
        and actual_series != expected_series
    ):

        raise ValueError(
            f"Model expects {expected_series} series "
            f"but deployment data has {actual_series}."
        )

    history_df = df

    print(
        f"Loaded {actual_series} series "
        f"and {len(df):,} observations."
    )


# -----------------------------------------------------
# Generate forecast once and cache it
# -----------------------------------------------------

def generate_all_forecasts():

    global forecast_df
    global model_column

    if forecast_df is not None:
        return forecast_df

    with forecast_lock:

        if forecast_df is not None:
            return forecast_df

        print("Running iTransformer inference...")

        predictions = nf.predict(
            df=history_df
        )

        if "unique_id" not in predictions.columns:
            predictions = predictions.reset_index()

        possible_columns = [
            c
            for c in predictions.columns
            if c not in {
                "unique_id",
                "ds",
                "cutoff"
            }
        ]

        if "iTransformer" in predictions.columns:
            model_column = "iTransformer"

        elif len(possible_columns) > 0:
            model_column = possible_columns[0]

        else:
            raise RuntimeError(
                "Could not find prediction column."
            )

        predictions[model_column] = (
            predictions[model_column]
            .clip(lower=0)
        )

        forecast_df = predictions[
            ["unique_id", "ds", model_column]
        ].copy()

        print(
            "Forecast generated:",
            len(forecast_df),
            "predictions"
        )

        return forecast_df


# -----------------------------------------------------
# Startup
# -----------------------------------------------------

@app.on_event("startup")
def startup():

    load_model()

    load_data()

    # You can comment this line if you want inference
    # to run only after the first user clicks Forecast.
    generate_all_forecasts()


# -----------------------------------------------------
# Health
# -----------------------------------------------------

@app.get("/api/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": nf is not None,
        "forecast_ready": forecast_df is not None,
        "series": (
            history_df["unique_id"].nunique()
            if history_df is not None
            else 0
        )
    }


# -----------------------------------------------------
# Model metrics
# -----------------------------------------------------

@app.get("/api/metrics")
def metrics():

    if not METRICS_PATH.exists():

        raise HTTPException(
            status_code=404,
            detail="metrics.json not found"
        )

    with open(
        METRICS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# -----------------------------------------------------
# Available series
# -----------------------------------------------------

@app.get("/api/series")
def get_series():

    ids = (
        history_df["unique_id"]
        .drop_duplicates()
        .tolist()
    )

    return {
        "count": len(ids),
        "series": [str(x) for x in ids]
    }


# -----------------------------------------------------
# Forecast selected series
# -----------------------------------------------------

@app.get("/api/forecast/{series_id}")
def get_forecast(series_id: str):

    predictions = generate_all_forecasts()

    history = history_df[
        history_df["unique_id"].astype(str)
        == str(series_id)
    ].copy()

    forecast = predictions[
        predictions["unique_id"].astype(str)
        == str(series_id)
    ].copy()

    if history.empty:

        raise HTTPException(
            status_code=404,
            detail="Series not found."
        )

    if forecast.empty:

        raise HTTPException(
            status_code=404,
            detail="Forecast not available."
        )

    history = history.sort_values("ds")

    forecast = forecast.sort_values("ds")

    predicted_total = float(
        forecast[model_column].sum()
    )

    predicted_average = float(
        forecast[model_column].mean()
    )

    peak_row = forecast.loc[
        forecast[model_column].idxmax()
    ]

    current_value = float(
        history.iloc[-1]["y"]
    )

    return {

        "series_id": str(series_id),

        "model": "iTransformer",

        "history": [
            {
                "ds": row["ds"].isoformat(),
                "y": float(row["y"])
            }
            for _, row in history.iterrows()
        ],

        "forecast": [
            {
                "ds": row["ds"].isoformat(),
                "yhat": float(
                    row[model_column]
                )
            }
            for _, row in forecast.iterrows()
        ],

        "summary": {

            "current_reviews_per_hour":
                current_value,

            "predicted_total_24h":
                predicted_total,

            "predicted_average_per_hour":
                predicted_average,

            "peak_prediction":
                float(
                    peak_row[model_column]
                ),

            "peak_time":
                peak_row["ds"].isoformat()
        }
    }


# -----------------------------------------------------
# Frontend
# -----------------------------------------------------

app.mount(
    "/static",
    StaticFiles(
        directory=str(FRONTEND_DIR)
    ),
    name="static"
)


@app.get("/")
def home():

    return FileResponse(
        FRONTEND_DIR / "index.html"
    )