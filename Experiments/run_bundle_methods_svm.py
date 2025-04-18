import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions.solver import bundle_svm_solver
from Functions.ampl_model import write_svm_ampl_data, run_ampl_svm
from Functions.plotting import (
    plot_convergence_comparison,
    plot_time_comparison,
    plot_accuracy_comparison,
    plot_objective_comparison
)


def generate_data(n_samples=200, n_features=5, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_informative=n_features,
        n_clusters_per_class=1,
        flip_y=0.05,
        class_sep=1.5,
        random_state=random_state
    )
    y = 2 * y - 1  # Convert to {-1, +1}
    return X, y

def load_iris_dataset(path="Datasets/adult.csv", binary=True):
    df = pd.read_csv(path)
    #df = df.sample(32000, random_state=42)  # Sample 1000 rows for faster processing

    if path == "Datasets/adult.csv":
        # Separate target
        y_raw = df["income"].values
        X = df.drop("income", axis=1)

        # Define categorical columns
        categorical_cols = ['Status', 'edu', 'mstatus', 'job', 'status2', 'ethni', 'country', 'gender']
        numerical_cols = [col for col in X.columns if col not in categorical_cols]

        # Apply OneHotEncoding to categorical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", numerical_cols),
                ("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ]
        )

        X_encoded = preprocessor.fit_transform(X)

    else:
        X = df.iloc[:, 1:-1].values  # Skip ID column
        y_raw = df.iloc[:, -1].values
        X_encoded = X

    # Binary label conversion
    if binary:
        class_labels = np.unique(y_raw)[:2]
        mask = np.isin(y_raw, class_labels)
        X_encoded = X_encoded[mask]
        y_raw = y_raw[mask]
        y = np.where(y_raw == class_labels[0], -1, 1)
    else:
        raise NotImplementedError("Multiclass not supported.")

    # Scaling and polynomial features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    poly = PolynomialFeatures(degree=1, include_bias=True)
    X_poly = poly.fit_transform(X_scaled)

    return X_encoded, y, X_poly, X_scaled


def run_experiments(use_dataset=False, dataset_path="Datasets/adult.csv",):

    # Step 1: Load data
    if use_dataset and os.path.exists(dataset_path):
        X, y, X_poly, X_scaled = load_iris_dataset(dataset_path)
        print(f"Loaded Iris dataset: {X.shape}")

        #encode categorical features like status, edu, ethinicity, gender. 
        #get all the columns names and their types[object, int, float]
        
    
    else:
        # Load synthetic dataset
        X, y = generate_data()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly.fit_transform(X_scaled)

    # Save to AMPL format
    write_svm_ampl_data(X_poly, y, filename="svm_data.dat", C=1.0)

   

    # === BUNDLE METHOD ===
    print("\n== Running Bundle Method ==")
    start = time.time()
    w_bundle, b_bundle, history_bundle = bundle_svm_solver(X_poly, y, C=1.0, step_size_strategy="line_search")
    bundle_time = time.time() - start
    y_pred_bundle = np.sign(X_poly @ w_bundle + b_bundle)
    acc_bundle = accuracy_score(y, y_pred_bundle)
    obj_bundle = history_bundle[-1]["f"]
    print("Bundle Objective:", obj_bundle)

    # === SKLEARN SVM ===
    print("\n== Running sklearn SVM ==")
    start = time.time()
    clf = SVC(kernel="poly", degree=2, C=1.0, gamma="auto", coef0=1)
    clf.fit(X_scaled, y)
    sklearn_time = time.time() - start
    acc_sklearn = accuracy_score(y, clf.predict(X_scaled))
    obj_sklearn = None  # Objective unavailable due to non-linear kernel

    # === AMPL SOLVER ===
    print("\n== Running AMPL Solver ==")
    try:
        start = time.time()
        w_ampl, b_ampl, xi_ampl = run_ampl_svm()

        ampl_time = time.time() - start
        y_pred_ampl = np.sign(X_poly @ w_ampl + b_ampl)
        acc_ampl = accuracy_score(y, y_pred_ampl)

        # Approximate primal objective: 0.5||w||^2 + C * ∑hinge_loss
        margins = y * (X_poly @ w_ampl + b_ampl)
        hinge_losses = np.maximum(0, 1 - margins)
        obj_ampl = 0.5 * np.dot(w_ampl, w_ampl) + 1.0 * np.sum(xi_ampl)

        print("Max diff between xi and hinge loss:", np.max(np.abs(xi_ampl - hinge_losses)))


        #obj_ampl = 0.5 * np.dot(w_ampl, w_ampl) + 1.0 * np.sum(hinge_losses)
    except Exception as e:
        print("AMPL failed:", e)
        acc_ampl = 0
        obj_ampl = None
        ampl_time = 0

    # === PLOTS ===
    plot_convergence_comparison([history_bundle], ["Bundle Method (Fixed)"])
    plot_accuracy_comparison(acc_bundle, acc_sklearn, acc_ampl)
    plot_time_comparison(bundle_time, sklearn_time, ampl_time)
    if obj_ampl is not None:
        plot_objective_comparison(obj_bundle, 0, obj_ampl)  # sklearn obj=0 as placeholder


if __name__ == "__main__":
    run_experiments( use_dataset=True, dataset_path="Datasets/adult.csv")
    #run_experiments(use_dataset=False)
