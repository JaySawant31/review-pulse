# ReviewPulse

### Deep Learning–Based Review Activity Forecasting and Deployment Platform

ReviewPulse is an end-to-end deep learning time-series forecasting application that predicts future hourly review activity using a trained **iTransformer** model.

The project uses the **Amazon Product and Google Locations Reviews** dataset and demonstrates a complete machine learning workflow — from data preprocessing and deep-learning model benchmarking to model selection, API development, interactive visualization, and Docker-based deployment.

---

## Overview

Online platforms can experience significant changes in review activity depending on time, product/location popularity, user behavior, and recurring temporal patterns.

ReviewPulse models these patterns to predict review activity for the **next 24 hours** using the previous **168 hours (7 days)** of observations.

Two modern Transformer-based forecasting architectures were trained and evaluated:

- **PatchTST**
- **iTransformer**

Both models were evaluated using the same train/validation/test setup. Based on the overall forecasting results, **iTransformer was selected as the final deployment model**.

The trained model is served through a **FastAPI backend**, visualized using an **HTML/CSS/JavaScript dashboard**, and packaged into a reproducible **Docker application**.

---

## Application Preview

<img width="2564" height="601" alt="image" src="https://github.com/user-attachments/assets/7545d20b-b1cf-48d9-8805-3c07aa5bc256" />

This application allows users to:

- Select one of the available review time-series IDs
- View the previous 168 hours of review activity
- Generate a 24-hour forecast
- Visualize historical and predicted review activity
- View predicted total review volume
- View average predicted hourly activity
- Identify predicted peak activity
- View the expected peak timestamp
- Inspect the deployed model's evaluation metrics

---

## Forecasting Problem

The forecasting task is formulated as:

```text
Previous 168 hours
        │
        ▼
  Deep Learning Model
        │
        ▼
Next 24-hour forecast
```

Mathematically:

$$
[y_{t-167}, \ldots, y_t]
\rightarrow
[\hat{y}_{t+1}, \ldots, \hat{y}_{t+24}]
$$

The deployed model receives the previous **168 hourly observations from all aligned time series** and predicts the following **24 hours of review activity**.

---

## Dataset

This project uses the:

**Amazon Product and Google Locations Reviews Dataset**

UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/1276/amazon+product+and+google+locations+reviews

The dataset used for training contains approximately:

- **3,055,000 observations**
- **235 aligned time series**
- **13,000 hourly observations per series**

### Main Variables

| Column | Description |
|---|---|
| `unique_id` | Identifier for an individual review time series |
| `ds` | Hourly timestamp |
| `y` | Number of reviews observed during that hour |

The original dataset is not stored in this GitHub repository because it exceeds GitHub's standard file-size limit.

For deployment, the application contains only the most recent **168 hours for each series**, stored in:

```text
data/latest_168h.csv
```

This produces:

```text
235 series × 168 hours = 39,480 deployment observations
```

---

## Data Preparation

The deployment-data preparation pipeline:

1. Loads the complete dataset
2. Removes unnecessary index columns
3. Converts timestamps into datetime format
4. Sorts observations by series and time
5. Selects the latest 168 common timestamps
6. Validates that every series contains exactly 168 observations
7. Saves the reduced inference dataset

The corresponding script is:

```text
prepare_deployment_data.py
```

---

## Model Development

Two deep learning architectures were trained independently using the same forecasting configuration.

### PatchTST

PatchTST applies Transformer attention to temporal patches extracted from a historical time-series window.

General flow:

```text
168-hour history
       │
       ▼
 Temporal Patches
       │
       ▼
Transformer Encoder
       │
       ▼
 Forecast Head
       │
       ▼
24-hour prediction
```

---

### iTransformer

iTransformer restructures Transformer-based forecasting by representing individual time-series variables as tokens.

For this project, the aligned review-activity series are processed jointly, allowing the model to learn temporal information as well as relationships across the different series.

```text
235 aligned review series
          │
          ▼
    iTransformer
          │
          ▼
Cross-series representation
          │
          ▼
24-hour forecasts
for all 235 series
```

---

## Model Evaluation

Both models were evaluated using the same forecasting setup:

- Input window: **168 hours**
- Forecast horizon: **24 hours**
- Validation period: **168 hours**
- Test period: **168 hours**
- Test strategy: **7 rolling 24-hour forecasts**
- Seasonal period: **168 hours**

The final test week therefore contains:

```text
235 series × 168 test hours
= 39,480 unseen predictions
```

A **weekly seasonal-naive model** was also used as a baseline.

The baseline predicts:

$$
\[
\hat{y}_t = y_{t-168}
\]
$$

In simple terms, it predicts an hour using the value observed at the same hour one week earlier.

---

## Model Performance

### Final Test Results

| Metric | PatchTST | iTransformer | Seasonal Naive |
|---|---:|---:|---:|
| MAE ↓ | 8.39 | **8.12** | 14.43 |
| RMSE ↓ | **32.62** | 34.27 | 42.02 |
| sMAPE ↓ | 62.03% | **58.62%** | 78.54% |
| WAPE ↓ | 19.02% | **18.39%** | 32.70% |
| MASE ↓ | 0.251 | **0.233** | 0.421 |

Lower values indicate better forecasting performance.

### Final Model Selection

**iTransformer** was selected for deployment because it achieved the best result on four of the five primary evaluation metrics:

- Lower MAE
- Lower MASE
- Lower WAPE
- Lower sMAPE

PatchTST achieved a slightly lower RMSE, indicating that it produced somewhat smaller errors on certain high-error observations.

Overall, iTransformer demonstrated better general forecasting performance across the complete set of time series.

---

## Improvement Over the Baseline

The weekly seasonal-naive model achieved:

```text
MAE = 14.43
```

while iTransformer achieved:

```text
MAE = 8.12
```

This corresponds to approximately a:

**44% reduction in average absolute forecasting error**

The WAPE also improved from:

```text
Seasonal Naive: 32.70%
iTransformer:   18.39%
```

indicating a substantial improvement in overall forecasting accuracy.

---

## Evaluation Metrics

### MAE — Mean Absolute Error

Measures the average absolute difference between predicted and actual review counts.

```text
iTransformer MAE = 8.12
```

The model is therefore off by approximately **8 reviews per hour on average** across the evaluated observations.

---

### RMSE — Root Mean Squared Error

Places greater emphasis on large forecasting errors.

```text
iTransformer RMSE = 34.27
```

---

### sMAPE — Symmetric Mean Absolute Percentage Error

Measures percentage-based forecasting error while handling low-valued observations better than standard MAPE.

```text
iTransformer sMAPE = 58.62%
```

---

### WAPE — Weighted Absolute Percentage Error

Measures overall absolute forecasting error relative to total actual review volume.

```text
iTransformer WAPE = 18.39%
```

---

### MASE — Mean Absolute Scaled Error

Normalizes model error relative to a seasonal-naive forecasting scale.

```text
iTransformer MASE = 0.233
```

This metric is particularly useful because the 235 time series can have substantially different review volumes.

---

## System Architecture

```text
             UCI Review Dataset
                     │
                     ▼
              Data Preprocessing
                     │
                     ▼
         Chronological Data Splitting
                     │
             ┌───────┴───────┐
             │               │
             ▼               ▼
         PatchTST       iTransformer
             │               │
             └───────┬───────┘
                     │
                     ▼
              Model Evaluation
                     │
                     ▼
          iTransformer Selected
                     │
                     ▼
           Saved Model Checkpoint
                     │
                     ▼
                 FastAPI
                     │
                     ▼
                REST API
                     │
                     ▼
          HTML + CSS + JavaScript
                     │
                     ▼
                 Chart.js
                     │
                     ▼
          Interactive Dashboard
                     │
                     ▼
                  Docker
```

For an even stronger repository presentation, a graphical version of this architecture can be added at:

```text
docs/images/architecture.png
```

and displayed using:

<!-- ![System Architecture](docs/images/architecture.png) -->

---

## Deployment Architecture

The trained model is **not retrained when the application starts or when a user requests a forecast**.

Instead:

```text
Saved iTransformer checkpoint
            │
            ▼
       FastAPI startup
            │
            ▼
Model loaded into memory
            │
            ▼
User selects a series
            │
            ▼
  Forecast API request
            │
            ▼
168-hour inference window
            │
            ▼
      iTransformer
            │
            ▼
  24-hour prediction
            │
            ▼
      JSON response
            │
            ▼
Interactive forecast graph
```

This separates the **training workflow** from the **production inference workflow**.

---

## Technology Stack

### Machine Learning

- Python
- PyTorch
- NeuralForecast
- PatchTST
- iTransformer
- Pandas
- NumPy

### Backend

- FastAPI
- Uvicorn

### Frontend

- HTML
- CSS
- JavaScript
- Chart.js

### Deployment

- Docker
- Docker Compose

### Development

- Git
- GitHub
- Visual Studio Code

---

## Repository Structure

```text
ReviewPulse/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   └── latest_168h.csv
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── model/
│   └── itransformer/
│       ├── alias_to_model.pkl
│       ├── configuration.pkl
│       ├── iTransformer_0.ckpt
│       ├── metrics.json
│       └── MODEL_PATH.txt
│
├── prepare_deployment_data.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
├── LICENSE
└── README.md
```

---

# Running ReviewPulse Locally

## Prerequisites

Install:

- Git
- Docker
- Docker Compose

Docker Desktop already includes Docker Compose on most modern installations.

You do not need to manually install the Python dependencies when using Docker.

---

## 1. Clone the Repository

```bash
git clone https://github.com/JaySawant31/ReviewPulse.git
```

Move into the project directory:

```bash
cd ReviewPulse
```

---

## 2. Build the Docker Image

Run:

```bash
docker compose build
```

Docker will install the required dependencies and package:

- FastAPI backend
- Frontend files
- iTransformer checkpoint
- Deployment dataset

---

## 3. Start the Application

Run:

```bash
docker compose up
```

Alternatively, build and run in one command:

```bash
docker compose up --build
```

When startup completes, the terminal should show output similar to:

```text
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

---

## 4. Open the Website

Open a browser and navigate to:

```text
http://localhost:8000
```

The ReviewPulse dashboard should now be available.

---

# Using the Website

## Step 1 — Select a Time Series

Use the dropdown menu to choose one of the available review-series IDs.

Example:

```text
Series 42
```

---

## Step 2 — Generate a Forecast

Click:

```text
Generate Forecast
```

The browser sends a forecast request to the FastAPI backend.

---

## Step 3 — Model Inference

The backend provides the previous:

```text
168 hours × 235 series
```

to the trained iTransformer.

The model predicts:

```text
24 hours × 235 series
```

The prediction associated with the selected series is then returned to the website.

---

## Step 4 — View the Forecast

The dashboard displays:

- Historical review activity
- iTransformer forecast
- Current hourly review activity
- Predicted total reviews over the next 24 hours
- Average predicted reviews per hour
- Expected peak hourly activity
- Expected peak time

---

## Example Forecast Visualization

A representative forecast screenshot can be added at:

```text
docs/images/forecast_example.png
```

Then display it with:

<!-- ![Example iTransformer Forecast](docs/images/forecast_example.png) -->

This figure is useful because it immediately demonstrates that the repository contains a working ML inference application rather than only model-training code.

---

# REST API

ReviewPulse exposes several REST API endpoints through FastAPI.

---

## Health Check

```http
GET /api/health
```

Browser:

```text
http://localhost:8000/api/health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "forecast_ready": true,
  "series": 235
}
```

---

## Available Series

```http
GET /api/series
```

Returns the list of available review-series IDs.

---

## Forecast

```http
GET /api/forecast/{series_id}
```

Example:

```http
GET /api/forecast/42
```

The response contains:

- Historical observations
- 24-hour predictions
- Current activity
- Predicted total
- Predicted hourly average
- Peak prediction
- Peak timestamp

---

## Model Metrics

```http
GET /api/metrics
```

Returns the stored evaluation metrics associated with the deployed iTransformer model.

---

# Interactive API Documentation

FastAPI automatically generates interactive API documentation.

After starting ReviewPulse, open:

```text
http://localhost:8000/docs
```

The Swagger interface can be used to directly inspect and test each API endpoint.

---

# Docker Commands

### Build the application

```bash
docker compose build
```

### Start the application

```bash
docker compose up
```

### Build and start

```bash
docker compose up --build
```

### Run in the background

```bash
docker compose up -d
```

### View logs

```bash
docker compose logs -f
```

### View running containers

```bash
docker ps
```

### Stop the application

```bash
docker compose down
```

---

## Saved Model

The trained deployment model is stored under:

```text
model/itransformer/
```

The model artifacts include the trained checkpoint and NeuralForecast configuration required to reconstruct the model for inference.

The trained model artifact is approximately:

```text
8.25 MB
```

making it lightweight enough to package directly with the application.

---

## Current Limitations

The current application performs forecasting using a fixed historical inference window stored in the repository.

Therefore, the application currently demonstrates **production-style inference on historical review data**, rather than continuously ingesting live Amazon or Google review activity.

Additional limitations include:

- Time series are represented by anonymized IDs rather than human-readable product or location names.
- The current model produces point forecasts rather than probabilistic prediction intervals.
- Real-time streaming ingestion is not currently implemented.
- The application currently uses a predefined 168-hour inference window.

---

## Future Improvements

Potential extensions include:

- Live review-data ingestion
- Real-time or scheduled forecasting
- Streaming pipelines
- PostgreSQL-based time-series storage
- Automated anomaly detection
- Probabilistic forecasting
- Confidence/prediction intervals
- Forecast calibration
- Model drift monitoring
- Data drift monitoring
- Scheduled model retraining
- GitHub Actions CI/CD
- Cloud deployment
- Public API hosting
- Human-readable product/location metadata
- Authentication and rate limiting
- Automated model-performance monitoring

---


## Key Project Outcomes

ReviewPulse demonstrates:

- Large-scale time-series preprocessing
- Deep learning for forecasting
- Transformer-based sequence modeling
- Multivariate forecasting
- Model benchmarking
- Temporal train/validation/test splitting
- Baseline comparison
- Model-selection methodology
- Model serialization
- REST API development
- Frontend/backend integration
- Interactive data visualization
- Docker containerization
- Reproducible deployment

---

## Project Summary

```text
3.05M Review Observations
          │
          ▼
     Data Pipeline
          │
   ┌──────┴──────┐
   │             │
PatchTST    iTransformer
   │             │
   └──────┬──────┘
          │
      Evaluation
          │
          ▼
   iTransformer
    MASE: 0.233
    MAE:  8.12
    WAPE: 18.39%
          │
          ▼
       FastAPI
          │
          ▼
    REST Interface
          │
          ▼
 HTML / CSS / JS
          │
          ▼
      Chart.js
          │
          ▼
        Docker
          │
          ▼
     ReviewPulse
```

---

## Author

**Jay Sunil Sawant**

Machine Learning / Artificial Intelligence

GitHub: [@JaySawant31](https://github.com/JaySawant31)

---

## License

This project is distributed under the license included in this repository.

See:

```text
LICENSE
```

for details.

---

## Acknowledgements

### Dataset

**Amazon Product and Google Locations Reviews**  
UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/1276/amazon+product+and+google+locations+reviews

### Open-Source Technologies

This project uses open-source tools and libraries including:

- PyTorch
- NeuralForecast
- FastAPI
- Uvicorn
- Chart.js
- Docker
- Pandas
- NumPy
