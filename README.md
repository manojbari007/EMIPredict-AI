# EMIPredict AI - Financial Risk Assessment

EMIPredict AI is an intelligent financial risk assessment platform designed to predict EMI eligibility and recommend the maximum feasible monthly EMI for an individual based on their demographic, financial, and credit profiles.

## Project Overview

The project aims to provide quick and reliable insights for lenders and financial institutions to make informed loan approval decisions, ensuring affordability and minimizing defaults.

### Key Features
- **EMI Eligibility Classification**: Uses classification models (Logistic Regression, Random Forest, XGBoost) to classify users into 'Eligible' or 'Not Eligible'.
- **Max Monthly EMI Prediction**: Uses regression models (Linear Regression, Random Forest, XGBoost) to predict the maximum affordable EMI amount.
- **Interactive UI**: A local Streamlit web app lets users input their financial details and receive instant real-time predictions.
- **Model Tracking**: Built-in support for MLflow to log model experiments, metrics, and parameters during training.

## Tech Stack
- **Data Manipulation**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost
- **Experiment Tracking**: mlflow
- **Web App**: streamlit
- **Serialization**: pickle

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/manojbari007/EMIPredict-AI.git
   cd EMIPredict-AI
   ```

2. **Install the dependencies**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Model Training (Optional)**
   The project comes with predefined models, but if you want to retrain them using your own `emi_prediction_dataset.csv` (not included due to size constraints), place the generic dataset in the root directory and run:
   ```bash
   python train.py
   ```
   This will train the models, track them using MLflow, and encode the `.pkl` files required by the Streamlit application.

4. **Launch the User Interface**
   Start the Streamlit risk assessment application:
   ```bash
   streamlit run app.py
   ```
   The application will start in your default web browser.

## Repository Structure
- `train.py`: Data loading, feature engineering, model training, and MLflow tracking script.
- `app.py`: Streamlit-based web application code for interactive predictions.
- `requirements.txt`: List of dependencies required to run the project.
- `*.pkl`: Serialized encoders, scalers, structural layouts, and the best classification/regression models.
- `extract.py` & `extracted_code.py` & `EMIPredict AI.ipynb`: Scripts & notebooks used for the initial model experimentation process.

---
*Created as part of a machine learning workflow demonstration.*
