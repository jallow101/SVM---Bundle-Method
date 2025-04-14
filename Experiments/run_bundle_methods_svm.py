import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import make_classification
from Functions.solver import bundle_svm_solver
from Functions.ampl_model import write_svm_ampl_data, run_ampl_svm
from Functions.plotting import plot_convergence, plot_accuracy_comparison, plot_time_comparison, plot_objective_comparison



def load_iris_dataset(path="Datasets/iris.csv", binary=True):
    df = pd.read_csv(path)
    X = df.iloc[:, 1:-1].values  # Skip ID column
    y_raw = df.iloc[:, -1].values

    if binary:
        class_labels = np.unique(y_raw)[:2]
        mask = np.isin(y_raw, class_labels)
        X = X[mask]
        y_raw = y_raw[mask]
        y = np.where(y_raw == class_labels[0], -1, 1)
    else:
        raise NotImplementedError("Multiclass not supported.")

    return X, y

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
    y = 2 * y - 1  # convert to {-1, +1}
    return X, y


acc_ampl = None

def run_experiment(use_dataset=False, dataset_path="Datasets/iris.csv",
                   degree=2, C=1.0, mu_0=1.0, tol=1e-4, n_samples=200, step_size_strategy="fixed"):

    # Step 1: Load data
    if use_dataset and os.path.exists(dataset_path):
        X, y = load_iris_dataset(dataset_path)
        print(f"Loaded Iris dataset: {X.shape}")
    else:
        X, y = generate_data(n_samples=n_samples)
        print(f" Generated synthetic dataset: {X.shape}")

    # Step 2: Normalize + expand
    scaler = StandardScaler()

    #convert string to categorical
    if isinstance(X, pd.DataFrame):
        X = X.apply(lambda col: pd.factorize(col)[0] if col.dtype == 'object' else col)
    elif isinstance(X, np.ndarray):
        X = pd.DataFrame(X).apply(lambda col: pd.factorize(col)[0] if col.dtype == 'object' else col).values
    else:
        raise ValueError("Unsupported data type for X.")
    

    X_scaled = scaler.fit_transform(X)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X_scaled)
    print(f" Polynomial feature shape: {X_poly.shape}")
    write_svm_ampl_data(X_poly, y, filename="svm_data.dat", C=C)
    print("AMPL data exported to svm_data.dat")


    # Step 3: Run primal SVM (bundle method)
    start_time = time.time()
    w_opt, b_opt, history = bundle_svm_solver(X_poly, y, C=C, mu_0=mu_0, tol=tol, step_size_strategy=step_size_strategy)
    end_time = time.time()
    bundle_time = end_time - start_time
   

    print("\n Final Results (Bundle Method)")
    print("Bias (b):", b_opt)
    print("Weight shape:", w_opt.shape)
    print("Final objective:", history[-1])
    #bundle method accuracy
    y_pred = np.sign(X_poly @ w_opt + b_opt)
    acc = accuracy_score(y, y_pred)
    print("Bundle Method Accuracy:", f"{acc * 100:.2f}%")
    acc_bundle = acc

    print(f"Time taken for Bundle Method: {bundle_time:.4f} seconds")

    # Step 4: Plot convergence

    plot_convergence(history, method_label="Bundle Method", degree=degree, strategy=step_size_strategy)
    
    
    # Step 5: Compare to sklearn SVC
    start_time = time.time()
    clf = SVC(kernel="poly", degree=degree, C=C, coef0=1, gamma="auto")  # gamma="auto" matches feature scaling
    clf.fit(X_scaled, y)
    y_pred = clf.predict(X_scaled)
    end_time = time.time()
    svc_time = end_time - start_time
    acc = accuracy_score(y, y_pred)

    print("\n Comparison: sklearn SVC")
    print("Support vectors:", clf.n_support_.sum())
    print("Accuracy on training data:", f"{acc * 100:.2f}%")
    acc_sklearn = acc

    print(f"Time taken for sklearn SVC: {svc_time:.4f} seconds")

    # Step 6: Step-Size Strategy Analysis (optional)
    if step_size_strategy == "fixed":
        print("Step-size strategy: Fixed (α = 2 / (2 + k))")
    elif step_size_strategy == "line_search":
        print("Step-size strategy: Line Search (optimal step-size)")
    else:
        print("Unknown step-size strategy")

    # Step 7: Run AMPL Solver
        

    try:
        w_ampl, b_ampl = run_ampl_svm()

        if len(w_ampl) == 0 or b_ampl is None:
            print("AMPL returned no valid solution.")
        else:
            y_pred_ampl = np.sign(X_poly @ w_ampl + b_ampl)
            acc_ampl = accuracy_score(y, y_pred_ampl)

            start_time = time.time()
            w_ampl, b_ampl = run_ampl_svm()
            ampl_time = time.time() - start_time
            

            print("\nAMPL Solver Results:")
            print(f"AMPL Accuracy: {acc_ampl * 100:.2f}%")
            print(f"AMPL Bias (b): {b_ampl}")
            print(f"AMPL Weights (first 5): {w_ampl[:5]}")

                
            plot_accuracy_comparison(acc_bundle, acc_sklearn, acc_ampl)
            plot_time_comparison(bundle_time, svc_time, ampl_time)


    except Exception as e:
        print("AMPL run failed:", e)

  
if __name__ == "__main__":
    run_experiment(use_dataset=True, step_size_strategy="fixed")
