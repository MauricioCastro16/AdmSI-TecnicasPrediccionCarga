import numpy as np
from typing import Sequence, Tuple

# --------- Datos (tu tabla) ---------
meses = ["Sep-2024","Oct-2024","Nov-2024","Dic-2024","Ene-2025","Feb-2025",
         "Mar-2025","Abr-2025","May-2025","Jun-2025","Jul-2025","Ago-2025"]
y = np.array([110, 120, 115, 70, 30, 40, 100, 112, 115, 120, 55, 105], dtype=float)

# x = 1..n (tiempo)
x = np.arange(1, len(y) + 1, dtype=float)

# --------- Fórmulas del apunte ---------
def linear_coeffs(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = len(x)
    x_bar, y_bar = x.mean(), y.mean()
    b = (np.sum(x*y) - n*x_bar*y_bar) / (np.sum(x**2) - n*(x_bar**2))
    a = y_bar - b*x_bar
    return float(a), float(b)

def predict(a: float, b: float, x: Sequence[float]) -> np.ndarray:
    return a + b * np.asarray(x, dtype=float)

def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))

# --------- Ajuste, ECM y pronósticos ---------
a, b = linear_coeffs(x, y)
y_hat = predict(a, b, x)
ecm_rl = mse(y, y_hat)

print(f"a (intercepto) = {a:.6f}")
print(f"b (pendiente)  = {b:.6f}")
print(f"ECM (Regresión Lineal) = {ecm_rl:.3f}")

# Pronóstico para los próximos 1 y 2 meses (Sep-2025 y Oct-2025)
f1 = predict(a, b, len(y) + 1)
f2 = predict(a, b, len(y) + 2)
print(f"Pronóstico próximo mes (t+1): {float(f1):.2f}")
print(f"Pronóstico siguiente (t+2): {float(f2):.2f}")
