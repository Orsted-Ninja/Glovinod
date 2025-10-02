import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- Configuration ---
FILE_PATH = "org.csv"
SKIP_ROWS = 53
TARGET_COL = 'koi_disposition'
FEATURE_COLS = [
    'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 'koi_depth', 'koi_prad',
    'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff', 'koi_slogg', 'koi_srad',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
]
MODEL_OUTPUT_PATH = "exoplanet_model_pipeline.joblib"

# --- Data Loading and Preprocessing ---
print("1. Loading and cleaning data...")
# Read CSV, skipping initial comment rows
df = pd.read_csv(FILE_PATH, skiprows=SKIP_ROWS)

# Drop all-null columns
df_clean = df.drop(columns=['koi_teq_err1', 'koi_teq_err2'])

X = df_clean[FEATURE_COLS]
y = df_clean[TARGET_COL]

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Identify feature types for the ColumnTransformer
numerical_features = [col for col in X.columns if 'fpflag' not in col]
flag_features = [col for col in X.columns if 'fpflag' in col]

# --- Build Pipeline ---
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('flag', 'passthrough', flag_features)
    ]
)

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# --- Training and Saving ---
print("2. Splitting data and training model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model_pipeline.fit(X_train, y_train)

# Save the model using your local library versions
print(f"3. Saving model to '{MODEL_OUTPUT_PATH}'...")
joblib.dump(model_pipeline, MODEL_OUTPUT_PATH)

print("\n🎉 Success! Model training and saving complete.")
print(f"The new model file ({MODEL_OUTPUT_PATH}) is now compatible with your system.")