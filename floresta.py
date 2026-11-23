# floresta aleatoria + RandomizedSearchCV
# depois, com o resto fixo no melhor valor, varia so o numero de arvores

import os
import time
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from plotnine import ggplot, aes, geom_line, geom_point, theme_bw, labs, scale_color_manual, ylim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

import ajuda

os.makedirs("figuras", exist_ok=True)
os.makedirs("resultados", exist_ok=True)

resultados = []
curvas = []

N_CURVA = [10, 25, 50, 100, 150, 200, 300, 400]


def roda_floresta(nome_dados, X, y, labels):
    print("\n######## FLORESTA -", nome_dados, "########")
    X_tr, X_te, y_tr, y_te = ajuda.split_padrao(X, y)
    print("treino", X_tr.shape[0], "teste", X_te.shape[0])

    param = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [4, 8, 12, 16, None],
        "max_features": ["sqrt", "log2", 0.5],
        "min_samples_leaf": [1, 2, 5, 10],
    }
    busca = RandomizedSearchCV(
        RandomForestClassifier(random_state=7, n_jobs=1),
        param,
        n_iter=12,
        cv=3,
        scoring="f1_weighted",
        random_state=7,
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    busca.fit(X_tr, y_tr)
    tempo = time.perf_counter() - t0
    print("melhor:", busca.best_params_, "cv:", round(busca.best_score_, 4), "tempo_s:", round(tempo, 1))

    extra = {
        "dataset": nome_dados,
        "cv": busca.best_score_,
        "params": str(busca.best_params_),
        "tempo_s": tempo,
    }

    print("treino:")
    m_tr = ajuda.metricas(y_tr, busca.predict(X_tr), nome=f"floresta_treino_{nome_dados}")
    m_tr.update(extra)
    m_tr["split"] = "treino"
    resultados.append(m_tr)

    print("teste:")
    m_te = ajuda.metricas(y_te, busca.predict(X_te), nome=f"floresta_teste_{nome_dados}")
    m_te.update(extra)
    m_te["split"] = "teste"
    resultados.append(m_te)
    print("gap f1 (treino - teste):", round(m_tr["f1"] - m_te["f1"], 4))

    ajuda.plot_matriz_hatch(
        m_te["cm"], labels, f"floresta {nome_dados} (teste)", f"figuras/floresta_cm_{nome_dados}.png"
    )
    ajuda.plot_circulo_confusao(
        m_te["cm"], labels, f"floresta {nome_dados}", f"figuras/floresta_circ_{nome_dados}.png"
    )

    # curva: os outros hiperparametros ficam no melhor valor
    fixos = {k: v for k, v in busca.best_params_.items() if k != "n_estimators"}
    print("curva n_estimators, fixos:", fixos)
    clf = RandomForestClassifier(warm_start=True, random_state=7, n_jobs=-1, **fixos)
    for n in N_CURVA:
        clf.set_params(n_estimators=n)
        clf.fit(X_tr, y_tr)
        f1_tr = ajuda.metricas(y_tr, clf.predict(X_tr), nome=f"rf_n{n}_treino")["f1"]
        f1_te = ajuda.metricas(y_te, clf.predict(X_te), nome=f"rf_n{n}_teste")["f1"]
        curvas.append({"dataset": nome_dados, "n_estimators": n, "f1_treino": f1_tr, "f1_teste": f1_te})

    ped = [c for c in curvas if c["dataset"] == nome_dados]
    melhor_f1 = max(c["f1_teste"] for c in ped)
    for c in ped:
        if melhor_f1 - c["f1_teste"] <= 0.005:
            print(
                "f1 de teste chega a 0.005 do maximo da curva em n =",
                c["n_estimators"],
                "(",
                round(c["f1_teste"], 4),
                "de",
                round(melhor_f1, 4),
                ")",
            )
            break

    longo = pd.DataFrame(ped).melt(
        id_vars="n_estimators",
        value_vars=["f1_treino", "f1_teste"],
        var_name="conjunto",
        value_name="f1",
    )
    longo["conjunto"] = longo["conjunto"].map({"f1_treino": "treino", "f1_teste": "teste"})
    ys = longo["f1"]
    margem = max(0.008, (ys.max() - ys.min()) * 0.35)
    (
        ggplot(longo, aes("n_estimators", "f1", color="conjunto"))
        + geom_line(size=1)
        + geom_point(size=2.5)
        + theme_bw()
        + scale_color_manual(values={"treino": "blue", "teste": "red"})
        + ylim(ys.min() - margem, ys.max() + margem)
        + labs(
            title=f"floresta {nome_dados}: F1 vs arvores",
            x="numero de arvores",
            y="F1",
            color="conjunto",
        )
    ).save(f"figuras/floresta_n_arvores_{nome_dados}.png", dpi=120, verbose=False)
    print("salvei", f"figuras/floresta_n_arvores_{nome_dados}.png")


X_cdc, y_cdc, _ = ajuda.carregar_cdc()
roda_floresta("cdc", X_cdc, y_cdc, ["0", "1"])

X_p, y_p, _, le = ajuda.carregar_pumpkin()
roda_floresta("pumpkin", X_p, y_p, list(le.classes_))

pd.DataFrame([ajuda.sem_cm(r) for r in resultados]).to_csv(
    "resultados/floresta_resumo.csv", index=False
)
pd.DataFrame(curvas).to_csv("resultados/floresta_n_arvores.csv", index=False)
print("ok floresta")
