"""V10b · Prueba unitaria: con recausalizar_locales=False, TEC queda sin dato 15 min después del último mapa."""
import sys; sys.path.insert(0, "src")
import pandas as pd
from s4 import features as ft
t = pd.date_range("2025-03-01 00:00", periods=120, freq="1min")
mapas = pd.Series([10.0, 11.0, 12.5, 12.0, 13.0], index=pd.date_range("2025-03-01 00:00", periods=5, freq="10min"))
integ = mapas.reindex(t.union(mapas.index)).ffill(limit=15).reindex(t)                 # como integrar_dataset v3
mv = pd.DataFrame({"TEC": integ, "ROTEC": integ, "ROTI": integ})
s4 = pd.DataFrame({"S4": 0.2, "reconstruido": False}, index=t)
for rec in (True, False):
    df = ft.construir_dataset(s4, mv, causal=True, verbose=False, ffill_max_min=0, recausalizar_locales=rec)
    ret = int((df.index.max() - mapas.index.max()) / pd.Timedelta("1min"))
    print(f"recausalizar_locales={rec!s:5s} -> último minuto con TEC: {ret} min después del último mapa")
df = ft.construir_dataset(s4, mv, causal=True, verbose=False, ffill_max_min=0, recausalizar_locales=False)
assert int((df.index.max() - mapas.index.max()) / pd.Timedelta("1min")) == 15, "La retención no es de 15 min"
print("OK: retención de 15 min")
