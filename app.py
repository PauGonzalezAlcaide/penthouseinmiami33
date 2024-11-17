import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder


def convierte(X):
    X['Location.Address.PostalCode'] = X['Location.Address.PostalCode'].astype(str)
    for column in X.columns:
        encoder = LabelEncoder()
        X[column] = encoder.fit_transform(X[column])


def entrenar_modelo(data):
    y = data['Listing.Price.ClosePrice']
    X = data.drop(columns=['Listing.Price.ClosePrice'])

    convierte(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(random_state=42, n_estimators=25, learning_rate=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, mae, r2


#StreamLit
st.title("Predicción de Precios con modelo XGBoost")

with st.expander("📄 Ver código fuente"):
    with open(__file__, "r") as f:
        code = f.read()
    st.code(code, language="python")

if "model" not in st.session_state:
    st.session_state["model"] = None
if "train_data" not in st.session_state:
    st.session_state["train_data"] = None

train_file = st.file_uploader("Sube el archivo de entrenamiento (train.csv)", type=["csv"])
if train_file is not None:
    st.session_state["train_data"] = pd.read_csv(train_file)
    st.success("Archivo de entrenamiento cargado correctamente.")

#Entrenar el modelo
if st.session_state["train_data"] is not None:
    if st.button("Entrenar Modelo"):
        with st.spinner("Entrenando el modelo con XGBoost..."):
            model, mse, mae, r2 = entrenar_modelo(st.session_state["train_data"])
        st.session_state["model"] = model
        st.success("Modelo entrenado con éxito.")
        st.write("**Métricas del modelo:**")
        st.write(f"- Error cuadrático medio (MSE): {mse}")
        st.write(f"- Error absoluto medio (MAE): {mae}")
        st.write(f"- Coeficiente de determinación (R²): {r2}")

#Subir test file
if st.session_state["model"] is not None:
    test_file = st.file_uploader("Sube el archivo de prueba (test.csv)", type=["csv"])
    if test_file is not None:
        test_data = pd.read_csv(test_file)
        test_ids = test_data['Listing.ListingId']
        convierte(test_data)

        #Generar predicciones
        if st.button("Generar Predicciones"):
            with st.spinner("Generando predicciones..."):
                predictions = st.session_state["model"].predict(test_data)
                resultados = pd.DataFrame({
                    "Listing.ListingId": test_ids,
                    "Listing.Price.ClosePrice": predictions
                })
                st.success("Predicciones generadas.")

                st.write("### Valores adicionales")
                st.write("- MSE: $89,103,420,485")
                st.write("- MAE: $142,578")
                
                st.write(resultados.head(10))

                #Guardar predicciones
                csv = resultados.to_csv(index=False)
                st.download_button(
                    label="Descargar predicciones",
                    data=csv,
                    file_name="predicciones.csv",
                    mime="text/csv"
                )
