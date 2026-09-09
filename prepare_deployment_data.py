import pandas as pd
from pathlib import Path


INPUT_FILE = Path("/Users/jaysawant/Desktop/ReviewPulse/data/y_amazon-google-large.csv")
OUTPUT_FILE = Path("/Users/jaysawant/Desktop/ReviewPulse/data/latest_168h.csv")

INPUT_SIZE = 168


print("Reading dataset...")

df = pd.read_csv(INPUT_FILE)

# Remove unnecessary exported index column
unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
if unnamed:
    df = df.drop(columns=unnamed)

df = df[["unique_id", "ds", "y"]].copy()

df["ds"] = pd.to_datetime(df["ds"])
df["y"] = pd.to_numeric(df["y"])

df = df.sort_values(["unique_id", "ds"])


# Find the latest 168 common timestamps
timestamps = sorted(df["ds"].unique())

latest_times = timestamps[-INPUT_SIZE:]


deployment_df = df[df["ds"].isin(latest_times)].copy()

deployment_df = deployment_df.sort_values(
    ["unique_id", "ds"]
).reset_index(drop=True)


# Validate
counts = deployment_df.groupby("unique_id").size()

print("Number of series:", deployment_df["unique_id"].nunique())
print("Rows per series:", counts.unique())
print("Total rows:", len(deployment_df))
print(
    "Date range:",
    deployment_df["ds"].min(),
    "->",
    deployment_df["ds"].max(),
)

if not (counts == INPUT_SIZE).all():
    raise ValueError(
        "Every series must contain exactly 168 observations."
    )


deployment_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)