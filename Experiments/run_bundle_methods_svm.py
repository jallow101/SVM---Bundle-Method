
import matplotlib.pyplot as plt
import os
import sys
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
<<<<<<< HEAD
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import make_classification
=======
import time
>>>>>>> ab6ee30d86b4b7364ff876d0a54796624dc67a6e

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

<<<<<<< HEAD
from Functions.solver import bundle_svm_solver


def load_iris_dataset(path="Datasets/iris.csv", binary=True):
=======
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import make_classification
from Functions.solver import bundle_svm_solver

def load_iris_dataset(path="Datasets/irzis.csv", binary=True):
>>>>>>> ab6ee30d86b4b7364ff876d0a54796624dc67a6e
    df = pd.read_csv(path)
    X = df.iloc[:, 1:-1].values
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

<<<<<<< HEAD

def generate_data(n_samples=200, n_features=2, random_state=42):
=======
def generate_data(n_samples=10000, n_features=20, random_state=42):
>>>>>>> ab6ee30d86b4b7364ff876d0a54796624dc67a6e
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

<<<<<<< HEAD

def run_experiment(use_dataset=True, dataset_path="Datasets/iris.csv",
                   degree=2, C=1.0, mu_0=1.0, tol=1e-4, n_samples=200):
=======
def run_experiment(use_dataset=False, dataset_path="Datasetzs/iris.csv",
                   degree=2, C=1.0, mu_0=1.0, tol=1e-4, n_samples=10000, step_size_strategy="line_search"):
>>>>>>> ab6ee30d86b4b7364ff876d0a54796624dc67a6e

    # Step 1: Load data
    if use_dataset and os.path.exists(dataset_path):
        X, y = load_iris_dataset(dataset_path)
        print(f"✅ Loaded Iris dataset: {X.shape}")
    else:
        X, y = generate_data(n_samples=n_samples)
        print(f"🧪 Generated synthetic dataset: {X.shape}")

    # Step 2: Normalize + expand
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X_scaled)
    print(f"✅ Polynomial feature shape: {X_poly.shape}")

    # Step 3: Run primal SVM (bundle method)
    start_time = time.time()
    w_opt, b_opt, history = bundle_svm_solver(X_poly, y, C=C, mu_0=mu_0, tol=tol, step_size_strategy=step_size_strategy)
    end_time = time.time()
    bundle_time = end_time - start_time

    # Step 4: Plot convergence
    objectives = [entry["f"] for entry in history]
    step_norms = [entry["step_norm"] for entry in history]

    plt.figure(figsize=(10, 5))
    plt.plot(objectives, marker='o', label="Objective Value")
    plt.plot(step_norms, marker='x', label="Step Norm")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Log-scale Value")
    plt.title(f"Convergence (Degree {degree})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/convergence_plot.png", dpi=300)
    plt.show()

    # Step 5: Compare to sklearn SVC
<<<<<<< HEAD
    clf = SVC(kernel="poly", degree=degree, C=C, coef0=1, gamma="auto")
=======
    start_time = time.time()
    clf = SVC(kernel="poly", degree=degree, C=C, coef0=1, gamma="auto")  # gamma="auto" matches feature scaling
>>>>>>> ab6ee30d86b4b7364ff876d0a54796624dc67a6e
    clf.fit(X_scaled, y)
    y_pred = clf.predict(X_scaled)
    end_time = time.time()
    svc_time = end_time - start_time

    # Accuracy
    acc = accuracy_score(y, y_pred)

    print("\n🎯 Final Results (Bundle Method)")
    print("Bias (b):", b_opt)
    print("Weight shape:", w_opt.shape)
<<<<<<< HEAD
    print("Final objective:", history[-1]["f"])
=======
    print("Final objective:", history[-1])
    print(f"Time taken for Bundle Method: {bundle_time:.4f} seconds")
>>>>>>> ab6ee30d86b4b7364ff876d0a54796624dc67a6e

    print("\n🤖 Comparison: sklearn SVC")
    print("Support vectors:", clf.n_support_.sum())
    print("Accuracy on training data:", f"{acc * 100:.2f}%")
    print(f"Time taken for sklearn SVC: {svc_time:.4f} seconds")

    # Step 6: Step-Size Strategy Analysis (optional)
    if step_size_strategy == "fixed":
        print("Step-size strategy: Fixed (α = 2 / (2 + k))")
    elif step_size_strategy == "line_search":
        print("Step-size strategy: Line Search (optimal step-size)")
    else:
        print("Unknown step-size strategy")


if __name__ == "__main__":
<<<<<<< HEAD
    run_experiment(use_dataset=False)
=======
    run_experiment(use_dataset=True, step_size_strategy="fixed")
>>>>>>> ab6ee30d86b4b7364ff876d0a54796624dc67a6e
