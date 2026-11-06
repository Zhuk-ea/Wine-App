import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model.joblib"
CONFIG_PATH = APP_DIR / "config.json"
TEST_PATH = APP_DIR / "test.csv"

LABELS = {
    "fixed_acidity": "Фиксированная кислотность",
    "volatile_acidity": "Летучая кислотность",
    "citric_acid": "Лимонная кислота",
    "residual_sugar": "Остаточный сахар",
    "chlorides": "Хлориды",
    "free_sulfur_dioxide": "Свободный SO2",
    "total_sulfur_dioxide": "Общий SO2",
    "density": "Плотность",
    "pH": "pH",
    "sulphates": "Сульфаты",
    "alcohol": "Алкоголь, %",
    "is_red": "Цвет",
}

STEPS = {
    "fixed_acidity": 0.1,
    "volatile_acidity": 0.01,
    "citric_acid": 0.01,
    "residual_sugar": 0.1,
    "chlorides": 0.001,
    "free_sulfur_dioxide": 1.0,
    "total_sulfur_dioxide": 1.0,
    "density": 0.0001,
    "pH": 0.01,
    "sulphates": 0.01,
    "alcohol": 0.1,
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test():
    return pd.read_csv(TEST_PATH)

@st.cache_data
def predict_proba_cached(_model, X):
    return _model.predict_proba(X)[:, 1]


if not MODEL_PATH.exists():
    st.error(
        f"Не найден {MODEL_PATH.name}"
    )
    st.stop()

model = load_model()
config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
default_threshold = float(config["threshold"])

test_df = load_test()
features = [c for c in test_df.columns if c != "target"]
X_test = test_df[features]
y_test = test_df["target"]
test_proba = predict_proba_cached(model, X_test)

with st.sidebar:
    st.header("Настройки")
    threshold = st.slider(
        "Порог хорошего вина",
        min_value=0.0,
        max_value=1.0,
        value=round(default_threshold, 2),
        step=0.01,
    )
    st.caption(f"В config.json: {default_threshold:.3f}")

st.title("Оценка качества вина")

tab_single, tab_metrics, tab_batch, tab_learned = st.tabs(["Одно вино", "Метрики модели", "Пакетное предсказание", "Что выучила модель"])

with tab_single:
    def feature_input(col_name: str, df: pd.DataFrame):
        if col_name == "is_red":
            median_red = int(round(float(df[col_name].median())))
            choice = st.radio(
                LABELS[col_name],
                ["Белое", "Красное"],
                index=median_red,
                horizontal=True,
            )
            return 1 if choice == "Красное" else 0

        lo = float(df[col_name].min())
        hi = float(df[col_name].max())
        med = float(df[col_name].median())
        if hi <= lo:
            hi = lo + 1.0
        step = STEPS.get(col_name, (hi - lo) / 100)
        return st.slider(LABELS[col_name], lo, hi, med, step=step)


    cols = st.columns(3)
    values = {}
    for i, col in enumerate(features):
        with cols[i % 3]:
            values[col] = feature_input(col, test_df)

    X_new = pd.DataFrame([values])[features]
    proba = float(model.predict_proba(X_new)[0, 1])

    st.markdown("---")
    left, right = st.columns([1, 2])
    with left:
        st.metric("Вероятность, что вино хорошее", f"{proba:.1%}")
    with right:
        st.progress(proba)
        if proba >= threshold:
            st.success(f"Хорошее вино")
        else:
            st.error(f"Плохое вино")

with tab_metrics:
    st.subheader("Метрики на тестовой выборке")
    st.write(f"Текущий порог: **{threshold:.2f}**")

    y_pred = (test_proba >= threshold).astype(int)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{metrics.precision_score(y_test, y_pred, zero_division=0):.3f}")
    m2.metric("Recall",    f"{metrics.recall_score(y_test, y_pred, zero_division=0):.3f}")
    m3.metric("F1",        f"{metrics.f1_score(y_test, y_pred, zero_division=0):.3f}")
    m4.metric("ROC-AUC",   f"{metrics.roc_auc_score(y_test, test_proba):.3f}")

    col_cm, col_txt = st.columns([1, 1])
    with col_cm:
        cm = metrics.confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
        metrics.ConfusionMatrixDisplay(cm, display_labels=["плохое", "хорошее"]).plot(
            ax=ax_cm, cmap="Blues", colorbar=False
        )
        ax_cm.set_title("Матрица ошибок")
        st.pyplot(fig_cm)
        plt.close(fig_cm)
    

    fpr, tpr, _ = metrics.roc_curve(y_test, test_proba)
    prec, rec, _ = metrics.precision_recall_curve(y_test, test_proba)

    tn, fp, fn, tp = cm.ravel()
    fpr_pt = fp / (fp + tn) if (fp + tn) else 0.0
    tpr_pt = tp / (tp + fn) if (tp + fn) else 0.0
    rec_pt = metrics.recall_score(y_test, y_pred, zero_division=0)
    prec_pt = metrics.precision_score(y_test, y_pred, zero_division=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {metrics.roc_auc_score(y_test, test_proba):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="случайная")
    axes[0].scatter([fpr_pt], [tpr_pt], color="crimson", s=90, zorder=5,
                    label=f"порог {threshold:.2f}")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC-кривая")
    axes[0].legend(loc="lower right")

    axes[1].plot(rec, prec, label=f"AP = {metrics.average_precision_score(y_test, test_proba):.3f}")
    axes[1].axhline(y_test.mean(), color="gray", linestyle="--", linewidth=1,
                    label="случайная")
    axes[1].scatter([rec_pt], [prec_pt], color="crimson", s=90, zorder=5,
                    label=f"порог {threshold:.2f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR-кривая")
    axes[1].legend(loc="upper right")

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Если порог увеличить, то модель меньше вин отмечает как хорошие: precision растёт, recall падает. В подборке меньше посредственных вин, но и хорошие вина чаще пропускаются.\n\n"
        "Если порог уменьшить будет наоборот: подборка шире, хороших вин в ней больше, но и посредственных тоже."
    )

with tab_batch:
    st.subheader("Пакетное предсказание")
    st.write(
        "Загрузите CSV с теми же признаками, что ожидает модель. "
    )

    uploaded = st.file_uploader("CSV с винами", type="csv")
    if uploaded is None:
        st.info("Ожидаемые признаки: " + ", ".join(features))
    else:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Не удалось прочитать CSV: {exc}")
        else:
            missing = [c for c in features if c not in batch.columns]
            if missing:
                st.error(f"В файле нет колонок: {', '.join(missing)}")
            else:
                proba = model.predict_proba(batch[features])[:, 1]
                result = batch.copy()
                result["wine_proba"] = proba
                result["wine_pred"] = proba >= threshold

                st.metric("Хороших вин по прогнозу", int(result["wine_pred"].sum()))
                st.dataframe(
                    result.sort_values("wine_proba", ascending=False),
                    hide_index=True,
                    use_container_width=True,
                )
                st.download_button(
                    "Скачать прогнозы",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="wine_predictions.csv",
                    mime="text/csv",
                    on_click="ignore",
                )

with tab_learned:
    st.subheader("Что выучила модель")
    st.write(
        "Коэффициенты логистической регрессии после StandardScaler. "
        "Положительные голосуют за хорошее вино, отрицательные - против."
    )

    coefs = model.named_steps["logisticregression"].coef_[0]
    coef_df = (
        pd.DataFrame({"feature": features, "coef": coefs})
        .sort_values("coef")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in coef_df["coef"]]
    ax.barh(coef_df["feature"], coef_df["coef"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Вклад признаков в решение хорошее ли вино")

    for i, val in enumerate(coef_df["coef"]):
        ax.text(
            val + (0.015 if val >= 0 else -0.015),
            i,
            f"{val:+.2f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
        )

    st.pyplot(fig)
    plt.close(fig)

    best_pos = coef_df.tail(3).iloc[::-1]["feature"].tolist()
    best_neg = coef_df.head(3)["feature"].tolist()
    st.markdown(
        "**Сильнее всего за хорошее вино:** " + ", ".join(f"`{f}`" for f in best_pos) + ".\n\n"
        "**Сильнее всего против:** " + ", ".join(f"`{f}`" for f in best_neg) + ".\n\n"
    )
