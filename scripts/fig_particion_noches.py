import sys; sys.path.insert(0, "src")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from s4.config import cargar_config, ruta
from s4.pipeline import preparar_datos

cfg = cargar_config("config/tesis_final.yaml")
datos = preparar_datos(cfg, verbose=False)
#noche = lambda s: s.resample("1D", offset="12h").max()          # noches de 07:00 a 07:00 hora de Lima
noche = lambda s: s.resample("24h", offset="12h").max()        # "24h" sí respeta el offset
diario = noche(datos["df"]["S4"])

fig, ax = plt.subplots(figsize=(14, 4.2))
colores = {"Entrenamiento": "#1F4E79", "Validación": "#B7791F", "Prueba": "#2E7D32"}
for (n, c), p in zip(colores.items(), [datos["train"], datos["val"], datos["test"]]):
    ax.axvspan(p.index.min(), p.index.max(), color=c, alpha=0.10)
    ev = int((noche(p["S4"]) >= 0.6).sum())
    ax.text(p.index.min() + (p.index.max() - p.index.min()) / 2, 1.32,
            f"{n}\n{p.index.min():%d/%m/%Y} – {p.index.max():%d/%m/%Y}\n{ev} noches con evento",
            ha="center", va="top", fontsize=10, color=c, fontweight="bold")
    print(f"{n}: {ev} noches con evento")
for p in [datos["val"], datos["test"]]:
    ax.axvline(p.index.min(), color="#444", lw=1.2, ls="--")
ax.plot(diario.index, diario.values, color="#333", lw=0.8, label="S4 máximo por noche")
e = diario[diario >= 0.6]
ax.scatter(e.index, e.values, s=10, color="#B23A48", zorder=3, label="Noche con evento (S4 ≥ 0.6)")
ax.axhline(0.6, color="#B23A48", ls=":", lw=1.2)
ax.set_ylim(0, 1.36); ax.set_ylabel("S4 máximo por noche")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.25)
ax.set_title("Partición cronológica Out-of-Time (corte a las 07:00 hora de Lima)", fontsize=12)
plt.tight_layout()
salida = ruta(cfg, "reportes") / "figures" / "fig_particion_oot_noches.png"
salida.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(salida, dpi=200)
print("Figura guardada en:", salida)