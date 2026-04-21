# ============================================================
# EARTHQUAKE PATTERN ANALYSIS
# Data Source: USGS Earthquake Catalog API
# BY: Srijita Kayal
# ============================================================
# --- STEP 1: Import Libraries ---

import requests
import pandas as pd

# --- STEP 1: Parameters ---
params = {
    "format": "csv",
    "starttime": "2024-01-01",
    "endtime": "2026-12-31",
    "minmagnitude": 5.0,
    "orderby": "time",
    "limit": 3000                  
}

# --- STEP 2: Fetch Data ---
url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
response = requests.get(url, params=params, timeout=30)
print("Status Code:", response.status_code)

with open("earthquakes.csv", "wb") as f:
    f.write(response.content)

# --- Load the Data ---
df = pd.read_csv("earthquakes.csv")
print(df.head())
print(df.shape)

# --- STEP 3: Data Cleaning ---

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nNull Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

# Convert time column to datetime
df['time'] = pd.to_datetime(df['time'])

# Extract useful time features
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day_of_week'] = df['time'].dt.day_name()

# Drop rows where magnitude or location is missing
df.dropna(subset=['mag', 'place', 'latitude', 'longitude'], inplace=True)

# Filter only actual earthquakes (remove quarry blasts etc.)
df = df[df['type'] == 'earthquake']

print("\nCleaned Shape:", df.shape)
print(df.head())

# --- STEP 4: EDA & Visualizations ---

import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("visuals", exist_ok=True)

# Plot 1: Magnitude Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['mag'], bins=30, kde=True, color='steelblue')
plt.title('Earthquake Magnitude Distribution (2023)')
plt.xlabel('Magnitude')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('visuals/magnitude_distribution.png')
plt.close()


# Plot 2: Depth vs Magnitude
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='depth', y='mag', alpha=0.4, color='purple')
plt.title('Depth vs Magnitude')
plt.xlabel('Depth (km)')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.savefig('visuals/depth_vs_magnitude.png')
plt.close()

# Plot 3: Monthly Frequency
plt.figure(figsize=(10, 4))
df['month'].value_counts().sort_index().plot(kind='bar', color='coral')
plt.title('Monthly Earthquake Frequency (2023)')
plt.xlabel('Month')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('visuals/monthly_frequency.png')
plt.close()

# Plot 4: Day of Week Distribution
plt.figure(figsize=(10, 4))
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['day_of_week'].value_counts().reindex(day_order).plot(kind='bar', color='teal')
plt.title('Earthquakes by Day of Week')
plt.xlabel('Day')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('visuals/day_of_week.png')
plt.close()

# STEP 5: STATISTICAL ANALYSIS
# ============================================================

# ---- 5A: Basic Stats ----
print("\n--- Magnitude Statistics ---")
print(df['mag'].describe())

print("\n--- Depth Statistics ---")
print(df['depth'].describe())

# ---- 5B: Depth Zone Classification ----
# Standard seismological classification
def classify_depth(d):
    if d < 70:
        return 'Shallow (<70km)'
    elif d < 300:
        return 'Intermediate (70-300km)'
    else:
        return 'Deep (>300km)'

df['depth_zone'] = df['depth'].apply(classify_depth)
print("\nDepth Zone Distribution:")
print(df['depth_zone'].value_counts())

# ---- 5C: Depth Zone vs Magnitude (Boxplot) ----
plt.figure(figsize=(10, 5))
zone_order = ['Shallow (<70km)', 'Intermediate (70-300km)', 'Deep (>300km)']
sns.boxplot(data=df, x='depth_zone', y='mag', order=zone_order, 
            hue='depth_zone', palette='Set2', legend=False)

plt.title('Magnitude Distribution by Depth Zone')
plt.xlabel('Depth Zone')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.savefig('visuals/depth_zone_magnitude.png')
plt.close()

# ---- 5D: Gutenberg-Richter Law ----
import numpy as np

mag_bins = np.arange(df['mag'].min(), df['mag'].max() + 0.1, 0.1)
counts = [(df['mag'] >= m).sum() for m in mag_bins]

plt.figure(figsize=(10, 5))
plt.plot(mag_bins, np.log10(counts), marker='o', markersize=3, color='darkred')
plt.title('Gutenberg-Richter Law (Log Frequency vs Magnitude)')
plt.xlabel('Magnitude (M)')
plt.ylabel('Log10(Cumulative Count)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/gutenberg_richter.png')
plt.close()

# ---- 5E: Correlation Heatmap ----
plt.figure(figsize=(10, 7))
numeric_cols = ['latitude', 'longitude', 'depth', 'mag', 'rms', 'gap', 'dmin']
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', 
            cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('visuals/correlation_heatmap.png')
plt.close()

print("\nAll statistical plots saved to visuals/")

# ============================================================
# STEP 6: INTERACTIVE WORLD MAP (FOLIUM)
# ============================================================
import folium

# Color by magnitude
def get_color(mag):
    if mag >= 7.0:
        return 'red'
    elif mag >= 6.0:
        return 'orange'
    elif mag >= 5.5:
        return 'yellow'
    else:
        return 'green'

# Create base map
eq_map = folium.Map(location=[0, 0], zoom_start=2, tiles='CartoDB dark_matter')

# Add each earthquake as a circle
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=row['mag'] * 1.5,
        color=get_color(row['mag']),
        fill=True,
        fill_opacity=0.6,
        popup=folium.Popup(
            f"<b>Place:</b> {row['place']}<br>"
            f"<b>Magnitude:</b> {row['mag']}<br>"
            f"<b>Depth:</b> {row['depth']} km<br>"
            f"<b>Date:</b> {str(row['time'])[:10]}",
            max_width=250
        )
    ).add_to(eq_map)

# Save as HTML
eq_map.save("visuals/earthquake_map.html")
print("Interactive map saved to visuals/earthquake_map.html")
# ============================================================
# STEP 7: MACHINE LEARNING — Magnitude Category Prediction
# ============================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import seaborn as sns

# ---- 7A: Create Target Variable ----
def classify_magnitude(mag):
    if mag < 5.5:
        return 'Moderate (5.0-5.5)'
    elif mag < 6.0:
        return 'Strong (5.5-6.0)'
    else:
        return 'Major/Great (6.0+)'

df['mag_category'] = df['mag'].apply(classify_magnitude)
print("\nMagnitude Category Distribution:")
print(df['mag_category'].value_counts())

# ---- 7B: Feature Selection ----
features = ['latitude', 'longitude', 'depth', 'rms', 'gap']
df_ml = df[features + ['mag_category']].dropna()

X = df_ml[features]
y = df_ml['mag_category']

# ---- 7C: Encode Target ----
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ---- 7D: Train/Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ---- 7E: Train Random Forest ----
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# ---- 7F: Evaluation ----
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, 
      target_names=le.classes_))
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc*100:.2f}%")

# Save cleaned ML-ready data for Power BI
df.to_csv("earthquakes_cleaned.csv", index=False)
print("Cleaned CSV saved for Power BI!")
# ---- 7G: Confusion Matrix ----
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Confusion Matrix — Magnitude Category Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('visuals/confusion_matrix.png')
plt.close()

# ---- 7H: Feature Importance ----
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(feat_df['Feature'], feat_df['Importance'], color='steelblue')
plt.title('Feature Importance — Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('visuals/feature_importance.png')
plt.close()

print("\nML visuals saved!")