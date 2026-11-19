"""
Ejecución reproducible de TODOS los experimentos de la tesis (E2–E6) sobre los
datos corregidos. Replica exactamente el diseño ya documentado (Tabla 3.6):
mismas variantes, mismas semillas y mismos valores de α y β en cada etapa.

Uso (desde la raíz del proyecto):
    python scripts/correr_experimentos.py --config config/tesis_final.yaml --etapa ventana
    python scripts/correr_experimentos.py --config config/tesis_final.yaml --etapa todas

Etapas (en este orden):
    ventana          E2  sensibilidad de la ventana (30–360 min), β = 100
    ablacion         E3  ablación principal (60 min), β = 100, con persistencia y climatología
    ablacion_extra   E3b A' (sin hora) y variantes de 180 min, β = 100
    alpha_beta       E4  fase 1: β × α, 1 semilla, umbral fijo
    beta             E4  fase 2: β ∈ {5, 10, 25, 50, 100}, α = 1.5, umbral calibrado
    alpha            E4  fase 3: α ∈ {1, 2}, β = 10
    patchtst         E5  LSTM, PatchTST y PatchTST sin RevIN + latencia
    precursores      E6  S4 + hora frente a precursores (60 y 180 min) + permutación de E
    explicabilidad   E6  permutación agrupada y gradientes integrados sobre D'

Todos los resultados van al registro registro_<nombre>.csv (grupos E2_…, E6_…).
El modelo final (E7, evaluación única en prueba) NO está aquí: se ejecuta aparte
con scripts/correr_modelo_final.py, después de revisar estas tablas en validación.
"""
import argparse, sys, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from s4.config import cargar_config, variante, ruta
from s4.pipeline import preparar_datos, correr_variante, baselines, construir_particiones, evaluar_y_registrar
from s4.registry import leer, resumir_semillas
from s4.train import configurar_gpu, medir_latencia
from s4.windows import inicios_comunes, a_escala_real
from s4 import features as ft, evaluate as ev, explain

WFL_E2_E3 = {"perdida.alpha": 1.5, "perdida.beta": 100.0}      # vigente cuando se hicieron E2 y E3
WFL_FINAL = {"perdida.alpha": 1.5, "perdida.beta": 10.0}       # resultado de E4
PATCH = {"tipo": "patchtst", "patch_len": 16, "stride": 8, "d_model": 64, "n_heads": 4, "n_capas": 2, "d_ff": 128, "dropout": 0.2}
MET = ("RMSE_global", "RMSE_evento", "RMSE_calma", "POD", "Precision", "HSS", "POD_inicio")
MET_OPT = ("HSS", "HSS_opt", "umbral_opt", "Precision_opt", "POD_inicio_opt", "RMSE_evento", "RMSE_global")


def tabla(cfg, grupo, metricas, nombre_csv):
    reg = leer(ruta(cfg, "experimentos") / f"registro_{cfg['nombre']}.csv", grupo=grupo, particion="val")
    t = resumir_semillas(reg, por=("modelo",), metricas=metricas)
    t.to_csv(ruta(cfg, "reportes") / "tables" / nombre_csv, index=False)
    print(f"\n=== {grupo} ===\n" + t.round(3).to_string(index=False), flush=True)
    return t


def permutacion(modelos, filas, parts, feats, grupos, u_fijo=None):
    """Importancia por permutación agrupada (3 repeticiones por grupo y semilla)."""
    p, esc = parts["val"], parts["escalador"]; m = ~p.objetivo_reconstruido
    X = p.X_lote(np.flatnonzero(m)); y, ult = p.y_real[m], p.ultimo_real[m]; fase = ev.fase_ventana(ult, y)
    rng = np.random.default_rng(0); out = []
    for (s, modelo, _), f in zip(modelos, filas):
        u = float(u_fijo if u_fijo is not None else f["umbral_opt"])
        pred = lambda Z: a_escala_real(modelo.predict(Z, batch_size=2048, verbose=0), esc)
        def met(yh):
            a = ev.metricas_alerta(y, yh, 0.6, u); ini = fase == "inicio"
            return a["HSS"], a["Precision"], float((yh[ini].max(axis=1) >= u).mean())
        b = met(pred(X))
        for g, cols in grupos.items():
            idx = [feats.index(c) for c in cols]
            for _ in range(3):
                Xp = X.copy(); Xp[:, :, idx] = X[rng.permutation(len(X))][:, :, idx]
                mm = met(pred(Xp)); del Xp
                out.append(dict(semilla=s, grupo=g, caida_HSS=b[0] - mm[0], caida_Precision=b[1] - mm[1], caida_POD_inicio=b[2] - mm[2]))
    r = pd.DataFrame(out).drop(columns="semilla").groupby("grupo").agg(["mean", "std"])
    r.columns = [f"{a}_{b}" for a, b in r.columns]
    return r.sort_values("caida_HSS_mean", ascending=False).reset_index(), X, fase


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config/tesis_final.yaml"); ap.add_argument("--etapa", required=True)
    a = ap.parse_args()
    etapas = ["ventana", "ablacion", "ablacion_extra", "alpha_beta", "beta", "alpha", "patchtst", "precursores", "explicabilidad"]
    pedir = etapas if a.etapa == "todas" else [a.etapa]
    configurar_gpu()
    base = cargar_config(a.config); datos = preparar_datos(base)
    TAB = ruta(base, "reportes") / "tables"; TAB.mkdir(parents=True, exist_ok=True)
    C = ft.CONJUNTOS

    if "ventana" in pedir:
        cfg = variante(base, **WFL_E2_E3); L = [30, 60, 120, 180, 240, 360]; H = cfg["ventanas"]["horizon"]
        com = {k: inicios_comunes(datos[k].index, L, H) for k in ("train", "val", "test")}
        for lb in L:
            correr_variante(variante(cfg, **{"ventanas.lookback": lb}), datos, C["B_S4_ciclica"], f"LB{lb}", "E2_ventana",
                            starts={k: com[k][lb] for k in com})
        tabla(base, "E2_ventana", MET, "E2_ventana.csv")

    if "ablacion" in pedir:
        cfg = variante(base, **WFL_E2_E3, **{"ventanas.lookback": 60})
        parts = construir_particiones(cfg, datos, C["A_S4"])
        for n, yh in baselines(cfg, datos, parts, "val").items():
            evaluar_y_registrar(cfg, parts["val"], yh, n, "E3_ablacion")
        for n, conj, ov in [("A · S4 + MSE", "A_S4", {"perdida.tipo": "mse"}), ("B · + hora + MSE", "B_S4_ciclica", {"perdida.tipo": "mse"}),
                            ("C · + hora + WFL", "B_S4_ciclica", {}), ("D · + TEC/ROTEC/ROTI", "D_locales", {}),
                            ("D' · + ROTI", "D_solo_ROTI", {}), ("E · + índices globales", "E_completo", {})]:
            correr_variante(variante(cfg, **ov), datos, C[conj], n, "E3_ablacion")
        tabla(base, "E3_ablacion", MET, "E3_ablacion.csv")

    if "ablacion_extra" in pedir:
        cfg = variante(base, **WFL_E2_E3)
        for n, conj, lb in [("A' · S4 + WFL", "A_S4", 60), ("C · + hora · LB180", "B_S4_ciclica", 180),
                            ("D' · + ROTI · LB180", "D_solo_ROTI", 180), ("E · completo · LB180", "E_completo", 180)]:
            correr_variante(variante(cfg, **{"ventanas.lookback": lb}), datos, C[conj], n, "E3b_ablacion_extra")
        tabla(base, "E3b_ablacion_extra", MET, "E3b_ablacion_extra.csv")

    if "alpha_beta" in pedir:
        cfg = variante(base, **{"ventanas.lookback": 180})
        for b in [25, 50, 100, 150, 200, 300]:
            for al in [1.0, 1.5, 2.0]:
                correr_variante(variante(cfg, **{"perdida.beta": float(b), "perdida.alpha": al}), datos, C["D_solo_ROTI"],
                                f"a{al}_b{b}", "E4_fase1", semillas=[42])
        tabla(base, "E4_fase1", MET, "E4_fase1.csv")

    if "beta" in pedir:
        cfg = variante(base, **{"ventanas.lookback": 180, "perdida.alpha": 1.5})
        for b in [5, 10, 25, 50, 100]:
            correr_variante(variante(cfg, **{"perdida.beta": float(b)}), datos, C["D_solo_ROTI"], f"b{b}", "E4_fase2", barrido_umbral=True)
        tabla(base, "E4_fase2", MET_OPT, "E4_fase2.csv")

    if "alpha" in pedir:
        cfg = variante(base, **{"ventanas.lookback": 180, "perdida.beta": 10.0})
        for al in [1.0, 2.0]:
            correr_variante(variante(cfg, **{"perdida.alpha": al}), datos, C["D_solo_ROTI"], f"a{al}_b10", "E4_fase3", barrido_umbral=True)
        tabla(base, "E4_fase3", MET_OPT, "E4_fase3.csv")

    if "patchtst" in pedir:
        cfg = variante(base, **WFL_FINAL, **{"ventanas.lookback": 180}); lat = []
        for n, c in [("LSTM apilada", cfg), ("PatchTST", variante(cfg, modelo=PATCH)), ("PatchTST sin RevIN", variante(cfg, modelo={**PATCH, "revin": False}))]:
            filas, mods, parts = correr_variante(c, datos, C["D_solo_ROTI"], n, "E5_patchtst", barrido_umbral=True, guardar_modelo=True)
            for disp in ["/CPU:0", "/GPU:0"]:
                lat.append({"modelo": n, **medir_latencia(mods[0][1], parts["val"], n=200, dispositivo=disp)})
            del mods
        pd.DataFrame(lat).to_csv(TAB / "E5_latencia.csv", index=False); print(pd.DataFrame(lat).round(2).to_string(index=False))
        tabla(base, "E5_patchtst", MET_OPT, "E5_patchtst.csv")

    if "precursores" in pedir:
        cfg = variante(base, **WFL_FINAL)
        for n, conj, lb in [("C · S4 + hora · LB60", "B_S4_ciclica", 60), ("C · S4 + hora · LB180", "B_S4_ciclica", 180),
                            ("D' · + ROTI · LB60", "D_solo_ROTI", 60), ("E · completo · LB60", "E_completo", 60),
                            ("D' · + ROTI · LB180", "D_solo_ROTI", 180), ("D · + TEC/ROTEC/ROTI · LB180", "D_locales", 180)]:
            correr_variante(variante(cfg, **{"ventanas.lookback": lb}), datos, C[conj], n, "E6_precursores", barrido_umbral=True)
        F = C["E_completo"]
        filas, mods, parts = correr_variante(variante(cfg, **{"ventanas.lookback": 180}), datos, F, "E · completo · LB180", "E6_precursores",
                                             barrido_umbral=True, guardar_modelo=True)
        G = {"S4": ["S4"], "Hora del día": ["daily_sin", "daily_cos"], "Día del año": ["day_of_year_sin", "day_of_year_cos"],
             "TEC y ROTEC": ["TEC", "ROTEC"], "ROTI": ["ROTI"], "Índices globales": ["Kp_Index", "Dst_Index", "AE_Index", "f10.7_Index"]}
        r, _, _ = permutacion(mods, filas, parts, F, {g: [c for c in cs if c in F] for g, cs in G.items()})
        r.to_csv(TAB / "E6_importancia_E.csv", index=False); print("\nIMPORTANCIA (E · 180 min)\n" + r.round(4).to_string(index=False))
        del mods
        tabla(base, "E6_precursores", MET_OPT, "E6_precursores.csv")

    if "explicabilidad" in pedir:
        F = C["D_solo_ROTI"]
        filas, mods, parts = correr_variante(variante(base, **WFL_FINAL, **{"ventanas.lookback": 180}), datos, F, "D' explicabilidad",
                                             "E6_explicabilidad", barrido_umbral=True, guardar_modelo=True)
        G = {"S4": ["S4"], "Hora del día": ["daily_sin", "daily_cos"], "Día del año": ["day_of_year_sin", "day_of_year_cos"], "ROTI": ["ROTI"]}
        r, X, fase = permutacion(mods, filas, parts, F, G, u_fijo=0.55)
        r.to_csv(TAB / "E6_importancia_Dprima.csv", index=False); print("\nIMPORTANCIA (D' · 180 min)\n" + r.round(4).to_string(index=False))
        ini = np.flatnonzero(fase == "inicio"); sel = np.random.default_rng(0).choice(ini, min(400, len(ini)), replace=False)
        at = np.concatenate([explain.gradientes_integrados(mods[0][1], X[sel[i:i + 100]], pasos=32) for i in range(0, len(sel), 100)])
        sal = pd.DataFrame(np.abs(at).mean(axis=0), columns=F); sal.insert(0, "minutos_antes_de_t", np.arange(X.shape[1] - 1, -1, -1))
        sal.to_csv(TAB / "E6_saliencia_inicio.csv", index=False)
        del mods

    print(f"\nETAPA(S) TERMINADA(S): {', '.join(pedir)}", flush=True)


if __name__ == "__main__":
    main()
