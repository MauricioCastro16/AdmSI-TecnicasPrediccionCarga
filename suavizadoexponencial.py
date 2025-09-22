import numpy as np

# -------- Datos (tu tabla) --------
y = np.array([110, 120, 115, 70, 30, 40, 100, 112, 115, 120, 55, 105], dtype=float)

# -------- Utilidades --------
def mse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean((y_true - y_pred) ** 2))

def ses_in_sample(y, alpha, f0=None):
    """
    Suavizado exponencial simple:
    f_t = f_{t-1} + α (y_{t-1} - f_{t-1}), t>=1
    Devuelve (serie f_t, f_{T+1})
    """
    y = np.asarray(y, float)
    f = np.zeros_like(y)
    f[0] = y[0] if f0 is None else float(f0)
    for t in range(1, len(y)):
        f[t] = f[t-1] + alpha * (y[t-1] - f[t-1])
    f_next = f[-1] + alpha * (y[-1] - f[-1])  # f_{T+1}
    return f, float(f_next)

def alpha_from_n(n, m=None):
    # Apunte: α = (2n-1)/(2n+1); extensión: α = (m n - 1)/(m n + 1), m>2
    return ((2*n - 1) / (2*n + 1)) if m is None else ((m*n - 1) / (m*n + 1))

# ===== 1) α fijo (barrido) =====
print("== SES con α fijo (barrido) ==")
alphas = np.linspace(0.05, 0.95, 19)  # 0.05, 0.10, ..., 0.95
scores = []
for a in alphas:
    f_in, f1 = ses_in_sample(y, a)
    ecm = mse(y[1:], f_in[1:])      # comparamos desde t=1 (f0 inicializado)
    scores.append((a, ecm, f1))
scores.sort(key=lambda t: t[1])

best_alpha, best_ecm, f1 = scores[0]
f2 = f1  # En SES simple, f_{T+h} = f_{T+1} para h>=1

print(f"Mejor α = {best_alpha:.2f} | ECM = {best_ecm:.3f}")
print(f"Pronóstico próximo mes (t+1): {float(f1):.2f}")
print(f"Pronóstico siguiente (t+2): {float(f2):.2f}")

# ===== 2) α(n) del apunte =====
print("\n== SES con α(n) del apunte ==")
by_n = []
for n in range(1, len(y)+1):
    a = alpha_from_n(n)            # o alpha_from_n(n, m=3) si querés más peso reciente
    f_in, f1 = ses_in_sample(y, a)
    ecm = mse(y[1:], f_in[1:])
    by_n.append((n, a, ecm, f1))
by_n.sort(key=lambda t: t[2])

best_n, a_n, ecm_n, f1_n = by_n[0]
f2_n = f1_n

print(f"Mejor n = {best_n} (α = {a_n:.6f}) | ECM = {ecm_n:.3f}")
print(f"Pronóstico próximo mes (t+1): {float(f1_n):.2f}")
print(f"Pronóstico siguiente (t+2): {float(f2_n):.2f}")
