import subprocess
import numpy as np

def write_svm_ampl_data(X, y, filename="svm_data.dat", C=1.0):
    n, d = X.shape
    with open(filename, "w") as f:
        f.write(f"param n := {n};\n")
        f.write(f"param d := {d};\n")
        f.write(f"param C := {C};\n\n")

        f.write("param y :=\n")
        for i in range(n):
            f.write(f"{i+1} {int(y[i])}\n")
        f.write(";\n\n")

        f.write("param X : " + " ".join(str(j+1) for j in range(d)) + " :=\n")
        for i in range(n):
            row = " ".join(f"{X[i, j]:.6f}" for j in range(d))
            f.write(f"{i+1} {row}\n")
        f.write(";\n")

def run_ampl_svm(mod="ampl/svm_model.mod", dat="svm_data.dat", sol="svm_solution.txt"):
    with open("ampl_script.run", "w") as f:
        f.write(f"""
model {mod};
data {dat};
option solver gurobi;
solve;
display w > {sol};
display b >> {sol};
display xi >> {sol};
""")

    # Run AMPL
    subprocess.run(["C:/Users/lenovo/OneDrive/Desktop/ampl_mswin64/ampl_mswin64/ampl.exe", "ampl_script.run"], check=True)

    # Parse solution
    w = []
    b = None
    reading_w = False

    with open(sol, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("w [*] :="):
                reading_w = True
                continue
            elif reading_w and ";" in line:
                reading_w = False
                continue
            elif reading_w:
                parts = line.split()
                for i in range(0, len(parts), 2):
                    w.append(float(parts[i + 1]))
            elif line.startswith("b ="):
                b = float(line.split("=")[-1])

    return np.array(w), b






if __name__ == "__main__":
    from ampl_model import run_ampl_svm
    w, b = run_ampl_svm()
    print("AMPL weights:", w[:5])
    print("AMPL bias:", b)
