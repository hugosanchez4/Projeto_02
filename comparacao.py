# compara o melhor xgboost com a melhor floresta (e a arvore, de referencia)
# le os csv gerados pelos outros scripts, nao treina de novo

import os
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_col,
    geom_text,
    position_dodge,
    theme_bw,
    labs,
    scale_fill_manual,
    facet_wrap,
    coord_cartesian,
)

import ajuda

os.makedirs("figuras", exist_ok=True)
os.makedirs("resultados", exist_ok=True)

arquivos = {
    "Arvore": "resultados/arvore_resumo.csv",
    "Floresta": "resultados/floresta_resumo.csv",
    "XGBoost": "resultados/xgb_resumo.csv",
}

partes = []
for nome, caminho in arquivos.items():
    if not os.path.exists(caminho):
        raise SystemExit(f"falta {caminho}. rode o script desse metodo antes.")
    df = pd.read_csv(caminho)
    df["metodo"] = nome
    partes.append(df)

tudo = pd.concat(partes, ignore_index=True)
teste = tudo[tudo["split"] == "teste"].copy()
treino = tudo[tudo["split"] == "treino"][["dataset", "metodo", "f1", "tempo_s"]].rename(
    columns={"f1": "f1_treino"}
)
tab = teste.merge(treino, on=["dataset", "metodo", "tempo_s"])
tab["gap_f1"] = tab["f1_treino"] - tab["f1"]
tab = tab.sort_values(["dataset", "metodo"])
tab.to_csv("resultados/comparacao.csv", index=False)
print(tab[["dataset", "metodo", "f1_treino", "f1", "gap_f1", "tempo_s"]].to_string(index=False))

longo = tab.melt(
    id_vars=["dataset", "metodo"],
    value_vars=["f1_treino", "f1"],
    var_name="conjunto",
    value_name="F1",
)
longo["conjunto"] = longo["conjunto"].map({"f1_treino": "treino", "f1": "teste"})
longo["rotulo"] = longo["F1"].map(lambda v: f"{v:.3f}")
(
    ggplot(longo, aes("metodo", "F1", fill="conjunto"))
    + geom_col(position=position_dodge(width=0.8), width=0.7)
    + geom_text(
        aes(label="rotulo"),
        position=position_dodge(width=0.8),
        va="bottom",
        size=7,
        nudge_y=0.006,
    )
    + facet_wrap("~dataset")
    + scale_fill_manual(values={"treino": "blue", "teste": "red"})
    + coord_cartesian(ylim=(0.78, 1.05))
    + theme_bw()
    + labs(title="F1 treino e teste", x="", y="F1", fill="conjunto")
).save("figuras/painel_comparacao.png", dpi=120, width=8, height=4, verbose=False)
print("salvei figuras/painel_comparacao.png")

for ds in ["cdc", "pumpkin"]:
    ped = (
        teste[teste["dataset"] == ds][["metodo", "acc", "prec", "rec", "f1"]]
        .rename(columns={"metodo": "modelo"})
        .reset_index(drop=True)
    )
    ajuda.plot_stream_metricas(ped, f"figuras/stream_{ds}.png")

print("\nfloresta vs xgboost:")
for ds in ["cdc", "pumpkin"]:
    ped = tab[tab["dataset"] == ds].set_index("metodo")
    diff = ped.loc["XGBoost", "f1"] - ped.loc["Floresta", "f1"]
    print(
        ds,
        "diff f1 (xgb - floresta):",
        round(diff, 4),
        "| tempo floresta:",
        round(ped.loc["Floresta", "tempo_s"], 1),
        "s | tempo xgb:",
        round(ped.loc["XGBoost", "tempo_s"], 1),
        "s",
    )
print("fim")
