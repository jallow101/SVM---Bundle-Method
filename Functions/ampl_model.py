# ampl_io.py
import subprocess
import numpy as np
import os

def write_ampl_data(file_path, G, F, x, mu):
    m, n = G.shape
    with open(file_path, 'w') as f:
        f.write(f"param n := {n};\nparam m := {m};\nparam mu := {mu};\n\n")
        f.write("param x :=\n" + "\n".join([f"{i+1} {x[i]}" for i in range(n)]) + ";\n\n")
        f.write("param f_const :=\n" + "\n".join([f"{j+1} {F[j]}" for j in range(m)]) + ";\n\n")
        f.write("param g :=\n")
        for j in range(m):
            for i in range(n):
                f.write(f"{j+1} {i+1} {G[j, i]}\n")
        f.write(";\n")

def run_ampl_solver(mod="ampl/bundle_master.mod", dat="bundle_data.dat", sol="ampl_out.txt"):
    with open("ampl_script.run", "w") as f:
        f.write(f"""
model {mod};
data {dat};
option solver cplex;
solve;
display d > {sol};
display v >> {sol};
""")
    subprocess.run(["ampl", "ampl_script.run"], check=True)

    d, v = [], None
    with open(sol, "r") as f:
        for line in f:
            if line.startswith("d["):
                d.append(float(line.split()[-1]))
            elif line.strip().startswith("v ="):
                v = float(line.strip().split("=")[-1])
    return np.array(d), v
