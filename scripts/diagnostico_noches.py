import sys; sys.path.insert(0, "src")
import pandas as pd
from s4.config import cargar_config
from s4.pipeline import preparar_datos
from s4.splits import dia_ionosferico

cfg = cargar_config("config/tesis_final.yaml")
datos = preparar_datos(cfg, verbose=False)
p = datos["test"]["S4"]

tabla = pd.Series(p.values, index=dia_ionosferico(p.index)).groupby(level=0).max()   # método de la Tabla 3.4
figura = p.resample("1D", offset="12h").max()                                         # método de la figura
nt, nf = set(tabla[tabla >= 0.6].index), set(figura[figura >= 0.6].index)
print("Noches con evento · tabla:", len(nt), "| figura:", len(nf))
for d in sorted(nf - nt) + sorted(nt - nf):
    ini = d + pd.Timedelta(hours=12)
    tramo = p.loc[ini: ini + pd.Timedelta(hours=24)]
    print(f"\nNoche que solo cuenta {'la figura' if d in nf else 'la tabla'}: {d.date()}")
    print("  S4 máximo:", round(tramo.max(), 3), "a las", tramo.idxmax(), "| minutos ≥ 0.6:", int((tramo >= 0.6).sum()))
    print("  primer y último minuto del tramo:", tramo.index.min(), "→", tramo.index.max())