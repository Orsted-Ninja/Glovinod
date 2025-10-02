# 🪐 Exoplanet Candidate Classifier

An interactive web application built with **Streamlit** that uses a **Random Forest Classifier** to predict the disposition of exoplanetary candidates based on transit survey data from the NASA Kepler mission.

The model is trained to classify a potential planetary signal as either **CONFIRMED**, a **CANDIDATE** requiring further review, or a **FALSE POSITIVE** (likely a stellar or instrumental artifact).

---

## 🚀 Getting Started

Follow these steps to set up and run the Exoplanet Classifier on your local machine.

### Prerequisites

1.  **Python:** Ensure you have Python 3.8+ installed.
2.  **Input Data:** You must have the NASA exoplanet dataset file (`org.csv`) placed in the root directory of this project.

### Installation

1.  **Clone/Download Project:** Get all the project files (`app.py`, `train_model.py`, `requirements.txt`) into a single folder.
2.  **Install Dependencies:** Open your terminal or command prompt in the project folder and run the following command to install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

---

## ⚙️ Usage

The project requires two steps: **Training the Model** and **Running the App**.

### Step 1: Train the Model

The model must be trained locally to ensure compatibility with your system's `scikit-learn` version. This script will read `org.csv` and save the trained pipeline as `exoplanet_model_pipeline.joblib`.

1.  Execute the training script in your terminal:

    ```bash
    python train_model.py
    ```

    A successful run will output: `The new model file (exoplanet_model_pipeline.joblib) is now compatible with your system.`

### Step 2: Run the Web App

Once the model file (`exoplanet_model_pipeline.joblib`) is in the same directory, you can launch the interactive Streamlit application.

1.  Execute the Streamlit command:

    ```bash
    streamlit run app.py
    ```

2.  Your default web browser should open automatically, displaying the app. You can now use the sliders and selectors in the sidebar to input data and get a real-time prediction from the model.

---

## 📊 Model Details

The machine learning pipeline consists of:

* **Model Type:** Random Forest Classifier
* **Preprocessing:**
    * **Imputation:** Missing numerical values are filled using the **median** of their respective columns.
    * **Scaling:** All features are normalized using **StandardScaler** (Z-score scaling).
* **Target Classes:**
    1.  `CANDIDATE` (Index 0)
    2.  `CONFIRMED` (Index 1)
    3.  `FALSE POSITIVE` (Index 2)