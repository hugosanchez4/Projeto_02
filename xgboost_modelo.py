# XGBoost + RandomizedSearchCV
# hiperparametros pedidos: n_estimators, learning_rate, max_depth, subsample

import os
import time
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

import ajuda

os.makedirs("figuras", exist_ok=True)
os.makedirs("resultados", exist_ok=True)

resultados = []


def roda_xgb(nome_dados, X, y, labels):
    print("\n######## XGBOOST -", nome_dados, "########")
    X_tr, X_te, y_tr, y_te = ajuda.split_padrao(X, y)
    print("treino", X_tr.shape[0], "teste", X_te.shape[0])

    param = {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.03, 0.1, 0.2],
        "max_depth": [3, 4, 6, 8],
        "subsample": [0.6, 0.8, 1.0],
    }
    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=7,
        n_jobs=1,
        verbosity=0,
    )
    busca = RandomizedSearchCV(
        base,
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
    m_tr = ajuda.metricas(y_tr, busca.predict(X_tr), nome=f"xgb_treino_{nome_dados}")
    m_tr.update(extra)
    m_tr["split"] = "treino"
    resultados.append(m_tr)

    print("teste:")
    m_te = ajuda.metricas(y_te, busca.predict(X_te), nome=f"xgb_teste_{nome_dados}")
    m_te.update(extra)
    m_te["split"] = "teste"
    resultados.append(m_te)
    print("gap f1 (treino - teste):", round(m_tr["f1"] - m_te["f1"], 4))

    ajuda.plot_matriz_hatch(
        m_te["cm"], labels, f"xgboost {nome_dados} (teste)", f"figuras/xgb_cm_{nome_dados}.png"
    )
    ajuda.plot_circulo_confusao(
        m_te["cm"], labels, f"xgboost {nome_dados}", f"figuras/xgb_circ_{nome_dados}.png"
    )


X_cdc, y_cdc, _ = ajuda.carregar_cdc()
roda_xgb("cdc", X_cdc, y_cdc, ["0", "1"])

X_p, y_p, _, le = ajuda.carregar_pumpkin()
roda_xgb("pumpkin", X_p, y_p, list(le.classes_))

pd.DataFrame([ajuda.sem_cm(r) for r in resultados]).to_csv(
    "resultados/xgb_resumo.csv", index=False
)
print("ok xgboost")
