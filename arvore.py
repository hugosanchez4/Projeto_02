# arvore de decisao + GridSearchCV
# hiperparametros pedidos: criterion e max_depth
# min_samples_leaf entra porque e o outro controle de memorizacao

import os
import time
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV

import ajuda

os.makedirs("figuras", exist_ok=True)
os.makedirs("resultados", exist_ok=True)

resultados = []


def roda_arvore(nome_dados, X, y, labels):
    print("\n######## ARVORE -", nome_dados, "########")
    X_tr, X_te, y_tr, y_te = ajuda.split_padrao(X, y)
    print("treino", X_tr.shape[0], "teste", X_te.shape[0])

    param = {
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 4, 6, 8, 12, None],
        "min_samples_leaf": [1, 5, 20],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=7),
        param,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    grid.fit(X_tr, y_tr)
    tempo = time.perf_counter() - t0
    print("melhor:", grid.best_params_, "cv:", round(grid.best_score_, 4), "tempo_s:", round(tempo, 1))

    extra = {
        "dataset": nome_dados,
        "cv": grid.best_score_,
        "params": str(grid.best_params_),
        "tempo_s": tempo,
    }

    print("treino:")
    m_tr = ajuda.metricas(y_tr, grid.predict(X_tr), nome=f"arvore_treino_{nome_dados}")
    m_tr.update(extra)
    m_tr["split"] = "treino"
    resultados.append(m_tr)

    print("teste:")
    m_te = ajuda.metricas(y_te, grid.predict(X_te), nome=f"arvore_teste_{nome_dados}")
    m_te.update(extra)
    m_te["split"] = "teste"
    resultados.append(m_te)
    print("gap f1 (treino - teste):", round(m_tr["f1"] - m_te["f1"], 4))

    ajuda.plot_matriz_hatch(
        m_te["cm"], labels, f"arvore {nome_dados} (teste)", f"figuras/arvore_cm_{nome_dados}.png"
    )
    ajuda.plot_circulo_confusao(
        m_te["cm"], labels, f"arvore {nome_dados}", f"figuras/arvore_circ_{nome_dados}.png"
    )

    # so o topo da arvore, senao o desenho fica ilegivel
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        grid.best_estimator_,
        max_depth=3,
        feature_names=list(X.columns),
        class_names=[str(c) for c in labels],
        filled=True,
        fontsize=7,
        ax=ax,
    )
    ax.set_title(f"arvore {nome_dados} (ate profundidade 3)")
    plt.tight_layout()
    plt.savefig(f"figuras/arvore_topo_{nome_dados}.png", dpi=120)
    plt.close()
    print("salvei", f"figuras/arvore_topo_{nome_dados}.png")


X_cdc, y_cdc, _ = ajuda.carregar_cdc()
roda_arvore("cdc", X_cdc, y_cdc, ["0", "1"])

X_p, y_p, _, le = ajuda.carregar_pumpkin()
roda_arvore("pumpkin", X_p, y_p, list(le.classes_))

pd.DataFrame([ajuda.sem_cm(r) for r in resultados]).to_csv(
    "resultados/arvore_resumo.csv", index=False
)
print("ok arvore")
