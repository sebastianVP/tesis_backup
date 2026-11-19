import pandas as pd
rec = pd.read_csv("data/processed/df_max_s4_reconstruido.csv", index_col=0, parse_dates=True).loc[:"2026-09-30 23:59"]
for f in ["Fix0", "Fix1", "Fix3", "Fix4"]:
    if f in rec.columns:
        rec[f] = rec[f].astype(str).str.lower().isin(["true", "1"])

# Cobertura diaria (minutos con dato / 1 440)
cob = rec.groupby(rec.index.normalize()).size() / 1440
print(f"Días en la serie final       : {len(cob)}")
print(f"Días con cobertura ≥ 90 %    : {(cob >= 0.90).sum()} ({100 * (cob >= 0.90).mean():.2f} %)")
print(f"Días con cobertura ≥ 98 %    : {(cob >= 0.98).sum()} ({100 * (cob >= 0.98).mean():.2f} %)")
print(f"Cobertura diaria mínima      : {100 * cob.min():.1f} %")

# Fix0: picos aislados corregidos
if "Fix0" in rec.columns:
    m = rec["Fix0"]
    episodios = int((m & ~m.shift(1, fill_value=False)).sum())
    print(f"\nFix0: {int(m.sum())} minutos en {episodios} episodios")
    print(rec.loc[m, ["S4"]].to_string())
else:
    print("\nLa columna Fix0 no está en el archivo")