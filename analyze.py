import pandas as pd, numpy as np, json, time

t0 = time.time()
PATH = "badata_ecobici_recorridos_realizados_2024.csv"

usecols = ["duracion_recorrido", "fecha_origen_recorrido", "nombre_estacion_origen",
           "modelo_bicicleta", "genero"]
dtypes = {"duracion_recorrido": "float64", "nombre_estacion_origen": "category",
          "modelo_bicicleta": "category", "genero": "category"}

df = pd.read_csv(r"C:\Especializacion\CEIA_Analisis_de_datos\Recorridos-en-Ecobicis\badata_ecobici_recorridos_realizados_2024.csv", usecols=usecols, dtype=dtypes, parse_dates=["fecha_origen_recorrido"])
print("loaded", df.shape, time.time() - t0)

out = {}
out["n_rows"] = int(len(df))
out["n_cols_original"] = 17

dur = df["duracion_recorrido"]
dur_min = dur / 60
out["dur_mean_min"] = float(dur_min.mean())
out["dur_median_min"] = float(dur_min.median())
out["dur_std_min"] = float(dur_min.std())
q1, q3 = dur_min.quantile([0.25, 0.75])
iqr = q3 - q1
mad = (dur_min - dur_min.median()).abs().median()
out["dur_q1_min"] = float(q1)
out["dur_q3_min"] = float(q3)
out["dur_iqr_min"] = float(iqr)
out["dur_mad_min"] = float(mad)
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
out["dur_outlier_pct"] = float(((dur_min < lo) | (dur_min > hi)).mean() * 100)
out["dur_na_pct"] = float(dur.isna().mean() * 100)

for col in ["genero", "modelo_bicicleta"]:
    vc = df[col].value_counts(normalize=True, dropna=False) * 100
    out[f"{col}_pct"] = {str(k): float(v) for k, v in vc.items()}

out["top_estaciones"] = {str(k): int(v) for k, v in df["nombre_estacion_origen"].value_counts().head(5).items()}

df["hora"] = df["fecha_origen_recorrido"].dt.hour
hora_counts = df["hora"].value_counts(normalize=True).sort_index() * 100
out["hora_pct"] = {str(int(k)): float(v) for k, v in hora_counts.items()}
top_horas = df["hora"].value_counts().head(3)
out["top_horas"] = {str(int(k)): int(v) for k, v in top_horas.items()}

df["dia_semana"] = df["fecha_origen_recorrido"].dt.day_name()
dia_counts = df["dia_semana"].value_counts(normalize=True) * 100
out["dia_pct"] = {str(k): float(v) for k, v in dia_counts.items()}

tab = pd.crosstab(df["modelo_bicicleta"], df["genero"], normalize="index") * 100
out["modelo_x_genero_pct"] = {str(idx): {str(c): float(v) for c, v in row.items()} for idx, row in tab.iterrows()}

with open("eda_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("saved eda_results.json", time.time() - t0)

# ---- Figures ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

p99 = dur_min.quantile(0.99)
plt.figure(figsize=(6, 4))
sns.histplot(dur_min[dur_min <= p99], bins=50, color="#1C7293")
plt.title("Duración del recorrido (min) - hasta P99")
plt.xlabel("minutos")
plt.tight_layout()
plt.savefig("fig_duracion_hist.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
order = df["hora"].value_counts().sort_index().index
sns.countplot(data=df, x="hora", color="#065A82")
plt.title("Recorridos por hora de inicio")
plt.xlabel("hora"); plt.ylabel("recorridos")
plt.tight_layout()
plt.savefig("fig_hora.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
vc = df["genero"].value_counts()
plt.pie(vc.values, labels=vc.index, autopct="%1.0f%%", colors=["#065A82", "#1C7293", "#9FD8CB"])
plt.title("Recorridos por género")
plt.tight_layout()
plt.savefig("fig_genero.png", dpi=150)
plt.close()

print("saved figures", time.time() - t0)
