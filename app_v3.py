import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ==================================================
# SZE STÍLUS
# ==================================================

st.set_page_config(
    page_title="AI Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

st.markdown("""
<style>

/* Háttér */
.stApp {
    background-color: #F7F9FB;
}

/* Címek */
h1, h2, h3 {
    color: #242943;
}

/* KPI kártyák */
[data-testid="stMetric"] {
    background: #EFF8FB;
    border-left: 6px solid #50ADC9;
    padding: 15px;
    border-radius: 10px;
}

/* Info doboz */
.stAlert {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# ADATOK BETÖLTÉSE
# ==================================================

df = pd.read_csv(
    "winequality-red.csv",
    sep=";"
)

# ==================================================
# MODELL TANÍTÁSA
# ==================================================

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_valid)

r2 = r2_score(y_valid, pred)

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

importance = pd.DataFrame({
    "Változó": X.columns,
    "Fontosság": model.feature_importances_
})

importance = importance.sort_values(
    by="Fontosság",
    ascending=False
)

# ==================================================
# OLDALSÓ MENÜ
# ==================================================

menu = st.sidebar.radio(
    "Navigáció",
    [
        "🏠 Főoldal",
        "📊 Adatvizsgálat",
        "🤖 Modell",
        "🍷 Predikció"
    ]
)

# ==================================================
# FŐOLDAL
# ==================================================

if menu == "🏠 Főoldal":

    st.title("🍷 AI Wine Quality Predictor")

    st.markdown("""
    ### Széchenyi István Egyetem

    **Csoportos Projektlabor**

    A projekt célja egy olyan mesterséges intelligencia modell létrehozása,
    amely a vörösborok fizikai és kémiai tulajdonságai alapján képes
    megbecsülni a várható minőséget.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Modell R² értéke",
            round(r2, 4)
        )

    with col2:
        st.metric(
            "Rekordok száma",
            len(df)
        )

        st.success("""
    ✅ Működő AI modell

    ✅ Random Forest Regressor

    ✅ 1599 rekordos adathalmaz

    ✅ Interaktív predikció
    """)

        st.info("""
    📊 Adathalmaz: UCI Wine Quality Dataset

    🤖 Modell: Random Forest Regressor

    🎯 Feladat:
    Bor minőségének előrejelzése
    fizikai és kémiai tulajdonságok alapján.
    """)

# ==================================================
# ADATVIZSGÁLAT
# ==================================================

elif menu == "📊 Adatvizsgálat":

    st.header("📊 Adatvizsgálat")

    st.subheader("A modell számára legfontosabb változók")

    st.bar_chart(
        importance.set_index("Változó")
    )

    st.subheader("Korrelációs hőtérkép")

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

# ==================================================
# MODELL
# ==================================================

elif menu == "🤖 Modell":

    st.header("🤖 AI Modell")

    st.metric(
        "R² érték",
        round(r2, 4)
    )

    st.write("""
### A modell működése

A rendszer Random Forest Regressor algoritmust használ.

A modell a bor fizikai és kémiai tulajdonságai alapján
próbálja megbecsülni a várható minőséget.

A tanítás a teljes, 1599 rekordos
Wine Quality adathalmazon történt.
""")

    st.subheader("Legfontosabb változók")

    st.dataframe(
        importance
    )

    st.subheader("🏆 A modell kedvenc bora")

    all_predictions = model.predict(X)

    best_pred_index = all_predictions.argmax()

    best_wine = df.iloc[[best_pred_index]]

    best_score = all_predictions[best_pred_index]

    st.success(
        f"A modell által becsült legjobb minőség: {best_score:.2f}"
    )

    st.dataframe(best_wine)

# ==================================================
# PREDIKCIÓ
# ==================================================

elif menu == "🍷 Predikció":

    st.header("🍷 Bor minőségének előrejelzése")

    fixed_acidity = st.slider(
        "Rögzített savtartalom",
        4.0, 16.0, 8.0
    )

    volatile_acidity = st.slider(
        "Illó savtartalom",
        0.1, 2.0, 0.5
    )

    citric_acid = st.slider(
        "Citromsav",
        0.0, 1.0, 0.3
    )

    residual_sugar = st.slider(
        "Maradék cukor",
        0.5, 16.0, 2.0
    )

    chlorides = st.slider(
        "Kloridok",
        0.01, 1.0, 0.08
    )

    free_sulfur_dioxide = st.slider(
        "Szabad kén-dioxid",
        1, 80, 15
    )

    total_sulfur_dioxide = st.slider(
        "Összes kén-dioxid",
        5, 300, 50
    )

    density = st.slider(
        "Sűrűség",
        min_value=0.990,
        max_value=1.005,
        value=0.996,
        step=0.0001,
        format="%.4f"
    )

    ph = st.slider(
        "pH",
        2.5, 4.5, 3.3
    )

    sulphates = st.slider(
        "Szulfátok",
        0.2, 2.0, 0.6
    )

    alcohol = st.slider(
        "Alkoholtartalom (%)",
        8.0, 15.0, 10.0
    )

    if st.button("🔮 Előrejelzés"):

        sample = pd.DataFrame({
            "fixed acidity": [fixed_acidity],
            "volatile acidity": [volatile_acidity],
            "citric acid": [citric_acid],
            "residual sugar": [residual_sugar],
            "chlorides": [chlorides],
            "free sulfur dioxide": [free_sulfur_dioxide],
            "total sulfur dioxide": [total_sulfur_dioxide],
            "density": [density],
            "pH": [ph],
            "sulphates": [sulphates],
            "alcohol": [alcohol]
        })

        quality = model.predict(sample)[0]

        st.success(
            f"Becsült minőség: {quality:.2f} / 10"
        )

        stars = round(quality)

        st.write("### Minősítés")

        st.write(
            "⭐" * stars +
            "☆" * (10 - stars)
        )

        if quality < 5:
            st.error("Gyenge minőségű bor")

        elif quality < 7:
            st.warning("Közepes minőségű bor")

        else:
            st.success("Jó minőségű bor")