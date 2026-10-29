import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --------------------------------------------------
# ADATOK BETÖLTÉSE
# --------------------------------------------------

df = pd.read_csv(
    "winequality-red.csv",
    sep=";"
)

# --------------------------------------------------
# MODELL TANÍTÁSA
# --------------------------------------------------

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

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.DataFrame({
    "Variable": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

# --------------------------------------------------
# OLDALSÓ MENÜ
# --------------------------------------------------

menu = st.sidebar.radio(
    "Menü",
    [
        "🏠 Főoldal",
        "📊 Adatvizsgálat",
        "🤖 Modell",
        "🍷 Predikció"
    ]
)

# --------------------------------------------------
# FŐOLDAL
# --------------------------------------------------

if menu == "🏠 Főoldal":

    st.title("🍷 AI Wine Quality Predictor")

    st.markdown("""
    ## Csoportos Projektlabor

    A rendszer a vörösborok minőségét becsüli meg
    a bor fizikai és kémiai tulajdonságai alapján.
    """)

    st.metric(
        "Modell R² értéke",
        round(r2, 4)
    )

    st.info(
        """
        📊 Adathalmaz: UCI Wine Quality Dataset

        📁 Rekordok száma: 1599

        🤖 Modell: Random Forest Regressor

        🎯 Feladat:
        Bor minőségének előrejelzése
        """
    )

# --------------------------------------------------
# ADATVIZSGÁLAT
# --------------------------------------------------

elif menu == "📊 Adatvizsgálat":

    st.header("📊 Adatvizsgálat")

    st.subheader("Változók fontossága")

    st.bar_chart(
        importance.set_index("Variable")
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

# --------------------------------------------------
# MODELL INFORMÁCIÓ
# --------------------------------------------------

elif menu == "🤖 Modell":

    st.header("🤖 AI Modell")

    st.metric(
        "R²",
        round(r2, 4)
    )

    st.write(
        """
        A projekt Random Forest Regressor modellt használ.

        A modell a bor minőségét próbálja előre jelezni
        a különböző kémiai jellemzők alapján.
        """
    )

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

# --------------------------------------------------
# PREDIKCIÓ
# --------------------------------------------------

elif menu == "🍷 Predikció":

    st.header("🍷 Bor minőség előrejelzése")

    fixed_acidity = st.slider(
        "Fixed Acidity",
        4.0,
        16.0,
        8.0
    )

    volatile_acidity = st.slider(
        "Volatile Acidity",
        0.1,
        2.0,
        0.5
    )

    citric_acid = st.slider(
        "Citric Acid",
        0.0,
        1.0,
        0.3
    )

    residual_sugar = st.slider(
        "Residual Sugar",
        0.5,
        16.0,
        2.0
    )

    chlorides = st.slider(
        "Chlorides",
        0.01,
        1.0,
        0.08
    )

    free_sulfur_dioxide = st.slider(
        "Free Sulfur Dioxide",
        1,
        80,
        15
    )

    total_sulfur_dioxide = st.slider(
        "Total Sulfur Dioxide",
        5,
        300,
        50
    )

    density = st.slider(
    "density",
    min_value=0.990,
    max_value=1.005,
    value=0.996,
    step=0.0001,
    format="%.4f"
    )


    ph = st.slider(
        "pH",
        2.5,
        4.5,
        3.3
    )

    sulphates = st.slider(
        "Sulphates",
        0.2,
        2.0,
        0.6
    )

    alcohol = st.slider(
        "Alcohol",
        8.0,
        15.0,
        10.0
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

        st.write("### Értékelés")

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