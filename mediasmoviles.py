import numpy as np

# --- Serie (tu tabla) ---
y = np.array([110, 120, 115, 70, 30, 40, 100, 112, 115, 120, 55, 105], dtype=float)

def moving_average_predictions(y, n):
    return np.array([y[i-n:i].mean() for i in range(n, len(y))])

def mse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))

def forecast_k_steps(y, n, k=2):
    vals = list(y)
    for _ in range(k):
        vals.append(np.mean(vals[-n:]))
    return vals[-k:]

# --- ECM para todas las n válidas ---
results = []
for n in range(2, len(y)):           # n = 2..11 (porque hay 12 datos)
    preds = moving_average_predictions(y, n)
    ecm = mse(y[n:], preds)
    results.append((n, ecm))

# Ranking (mejor a peor)
ranking = sorted(results, key=lambda t: t[1])
best_n, best_ecm = ranking[0]

print("\nRanking (mejor→peor):")
for n, e in ranking:
    print(f"  n={n:2d}  ECM={e:,.3f}")

# Pronósticos con la mejor n
f1, f2 = forecast_k_steps(y, best_n, k=2)
print(f"\nMejor n (media móvil): {best_n}  |  ECM = {best_ecm:.3f}")
print(f"Pronóstico próximo mes (t+1): {float(f1):.2f}")
print(f"Pronóstico siguiente (t+2): {float(f2):.2f}")
