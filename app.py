import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# -----------------------------
# Adatok betöltése
# -----------------------------

df = pd.read_csv(
    "winequality-red.csv",
    sep=";"
)

# -----------------------------
# Modell tanítás
# -----------------------------

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
importance = pd.DataFrame({
    "Variable": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)
st.subheader("Korrelációs hőtérkép")

fig, ax = plt.subplots(
    figsize=(10,8)
)

sns.heatmap(
    df.corr(),
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)


pred = model.predict(X_valid)

r2 = r2_score(y_valid, pred)
st.subheader("Legfontosabb tényezők")

st.bar_chart(
    importance.set_index("Variable")
)


# -----------------------------
# Streamlit felület
# -----------------------------

menu = st.sidebar.radio(
    "Menü",
    [
        "🏠 Főoldal",
        "📊 Adatvizsgálat",
        "🤖 Modell",
        "🍷 Predikció"
    ]
)
if menu == "🏠 Főoldal":
   st.title("🍷 AI Wine Quality Predictor")
   st.markdown("""
   # 🍷 AI Wine Quality Predictor

   ### Csoportos Projektlabor

   A rendszer a vörösborok minőségét becsüli meg
   a bor kémiai jellemzői alapján.

   A modell teljesítménye:

   ✅ Random Forest Regressor

   ✅ 1599 rekordos adathalmaz

   ✅ R² = 0.5319
   """)
if menu == "📊 Adatvizsgálat":

    st.header("📊 Adatvizsgálat")
if menu == "🤖 Modell":

    st.header("🤖 Modell")

    st.metric(
        "Modell R² értéke",
        round(r2, 4)
    )

    st.info(
        """
        Adathalmaz: UCI Wine Quality Dataset

        Rekordok száma: 1599

        Modell: Random Forest Regressor

        Cél: A bor minőségének előrejelzése.
        """
    )

    st.subheader("Legfontosabb tényezők")

    st.bar_chart(
        importance.set_index("Variable")
    )

    st.subheader("Korrelációs hőtérkép")

    fig, ax = plt.subplots(figsize=(10,8))

    sns.heatmap(
        df.corr(),
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)
if menu == "🍷 Predikció":

    st.write(
    "Bor minőségének előrejelzése mesterséges intelligencia segítségével"
    )

    st.metric(
    "Modell R² értéke",
    round(r2, 4)
    )
    st.info(
    """
    Adathalmaz: UCI Wine Quality Dataset
    
    Rekordok száma: 1599
    
    Modell: Random Forest Regressor
    
    Cél: A bor minőségének előrejelzése
    fizikai és kémiai tulajdonságok alapján.
    """
    )

    st.divider()

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
    "Density",
    0.990,
    1.005,
    0.996
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
        st.warning("Gyenge minőség")

    elif quality < 7:
        st.info("Közepes minőség")

    else:
        st.success("Jó minőség")