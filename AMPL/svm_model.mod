param n;
param d;
param C;

param X {1..n, 1..d};
param y {1..n};

var w {1..d};
var b;
var xi {1..n} >= 0;

minimize Objective:
    0.5 * sum {j in 1..d} w[j]^2 + C * sum {i in 1..n} xi[i];

subject to Margin {i in 1..n}:
    y[i] * (sum {j in 1..d} w[j] * X[i,j] + b) >= 1 - xi[i];
