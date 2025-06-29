import streamlit as st
import yaml
import bcrypt
import os
import yagmail
from dotenv import load_dotenv
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import timedelta
import statsmodels.api as sm
from arima import load_data_set, build_model_predict_arima, forecast_future_days, evaluate_performance_arima, plot_arima
from rnn import rnn_model

# Load environment variables (for email credentials)
load_dotenv()
EMAIL = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# --- Helper functions for user data management ---
def load_users():
    try:
        if not os.path.exists("users.yaml"):
            with open("users.yaml", "w") as f:
                yaml.dump({"credentials": {"usernames": {}}}, f)
            return {"credentials": {"usernames": {}}}

        with open("users.yaml", "r") as f:
            data = yaml.safe_load(f)
            if data is None:
                return {"credentials": {"usernames": {}}}
            if "credentials" not in data or "usernames" not in data["credentials"]:
                data["credentials"] = {"usernames": {}}
            return data
    except Exception as e:
        st.error(f"Error loading user data: {e}. Please check 'users.yaml'.")
        return {"credentials": {"usernames": {}}}

def save_users(users):
    try:
        with open("users.yaml", "w") as f:
            yaml.dump(users, f, default_flow_style=False)
        st.success("User data saved successfully.")
    except Exception as e:
        st.error(f"Error saving user data: {e}")

# --- Initialize session state variables ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Acasă"
if "username" not in st.session_state:
    st.session_state["username"] = None

# --- Streamlit config ---
st.set_page_config(page_title="Autentificare și Navigare", layout="centered")
st.title("🔐 Aplicație de Autentificare")

# --- Autentificare și înregistrare ---
if not st.session_state["logged_in"]:
    menu = st.sidebar.selectbox("Alege acțiunea:", ["Login", "Register", "Forgot Password"], key="auth_menu_app")
    users_data = load_users()

    if menu == "Register":
        st.subheader("🆕 Înregistrare")
        new_username = st.text_input("Nume utilizator", key="reg_username_app")
        new_email = st.text_input("Email", key="reg_email_app")
        new_password = st.text_input("Parolă", type="password", key="reg_password_app")

        if st.button("Creează cont", key="create_account_btn_app"):
            if not new_username or not new_email or not new_password:
                st.error("Toate câmpurile sunt obligatorii.")
            elif new_username in users_data["credentials"]["usernames"]:
                st.error("Acest utilizator există deja.")
            else:
                hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                users_data["credentials"]["usernames"][new_username] = {
                    "email": new_email,
                    "name": new_username,
                    "password": hashed_pw,
                    "profile_image": None,
                    "notifications": False
                }
                save_users(users_data)
                st.success("Cont creat cu succes! Te poți autentifica acum.")
                st.session_state["auth_menu_app"] = "Login"
                st.rerun()

    elif menu == "Login":
        st.subheader("🔑 Autentificare")
        username_input = st.text_input("Nume utilizator", key="login_username_app")
        password_input = st.text_input("Parolă", type="password", key="login_password_app")

        if st.button("Login", key="login_btn_app"):
            if username_input in users_data["credentials"]["usernames"]:
                hashed_pw = users_data["credentials"]["usernames"][username_input]["password"]
                if bcrypt.checkpw(password_input.encode(), hashed_pw.encode()):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username_input
                    st.success(f"Bine ai venit, {username_input}!")
                    st.rerun()
                else:
                    st.error("Parolă greșită.")
            else:
                st.error("Utilizatorul nu există.")

    elif menu == "Forgot Password":
        st.subheader("🔁 Recuperare parolă")
        email_forgot_password = st.text_input("Introduceți emailul asociat", key="forgot_password_email_app")
        if st.button("Trimite email", key="send_email_btn_app"):
            found_email = False
            for user, info in users_data["credentials"]["usernames"].items():
                if info.get("email") == email_forgot_password:
                    found_email = True
                    try:
                        if not EMAIL or not EMAIL_PASSWORD:
                            st.error("Credențialele de email lipsesc în .env.")
                            break
                        yag = yagmail.SMTP(user=EMAIL, password=EMAIL_PASSWORD)
                        yag.send(to=email_forgot_password, subject="Recuperare parolă",
                                 contents=f"Salut, {info['name']}! Contactează adminul pentru resetare.")
                        st.success("Email trimis cu succes!")
                    except Exception as e:
                        st.error(f"Eroare la email: {str(e)}")
                    break
            if not found_email:
                st.error("Emailul nu a fost găsit.")



# --- Dashboard după autentificare ---
else:
    st.title("📊 Dashboard principal")
    st.success(f"Ești autentificat ca: **{st.session_state['username']}**")

    valid_pages = ["Acasă", "Setări", "Predictie ARIMA", "Predictie RNN", "Delogare"]
    if "pagina_selectata_app" not in st.session_state or st.session_state["pagina_selectata_app"] not in valid_pages:
        st.session_state["pagina_selectata_app"] = "Acasă"

    st.selectbox("Navighează către:", valid_pages,
                 index=valid_pages.index(st.session_state["pagina_selectata_app"]),
                 key="pagina_selectata_app")

    st.session_state["current_page"] = st.session_state["pagina_selectata_app"]
    current_page = st.session_state["current_page"]

    if current_page == "Acasă":
        st.subheader("🏠 Pagina Acasă")
    
    elif current_page == "Predictie ARIMA":
        st.subheader("📈 Predicție ARIMA pentru curs valutar")

        uploaded_file = st.file_uploader("Încarcă fișierul CSV cu date valutare", type="csv")

        if uploaded_file is not None:
            with open("currency_prediction_data_set.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())

            df = pd.read_csv("currency_prediction_data_set.csv", index_col=0)
            st.write("Date încărcate:", df.head())

            valuta_selectata = st.selectbox("Alege valuta pentru predicție:", [col[4:] for col in df.columns])
            predict_type = st.radio("Selectează tipul de predicție:", ["Pentru mai multe zile", "Pentru o zi anume"])

            if predict_type == "Pentru mai multe zile":
                forecast_days = st.slider("Număr de zile de prezis:", min_value=1, max_value=30, value=10)
            else:
                ultima_zi = pd.to_datetime(df.index[-1])
                data_selectata = st.date_input("Alege data țintă pentru predicție:", min_value=ultima_zi + timedelta(days=1))
                forecast_days = (data_selectata - ultima_zi.date()).days

            if st.button("Execută predicția ARIMA"):
                try:
                    raw_data, dates = load_data_set(valuta_selectata)
                    training_actual, testing_actual = raw_data[:-forecast_days], raw_data[-forecast_days:]
                    arima_model_fit, testing_predict, training_predict_series = build_model_predict_arima(training_actual, testing_actual)

                    st.write("📊 Rezultatele predicției:")
                    forecast_df = pd.DataFrame({
                        "Data": pd.date_range(start=dates[-1] + timedelta(days=1), periods=forecast_days),
                        "Valoare prezisă": forecast_future_days(arima_model_fit, training_predict_series, forecast_days)
                    })
                    st.dataframe(forecast_df)

                    plot_arima(valuta_selectata, testing_actual, testing_predict, "predictie_arima.pdf", training_series=training_actual)
                    with open("predictie_arima.pdf", "rb") as f:
                        st.download_button("📥 Descarcă graficul PDF", data=f, file_name="predictie_arima.pdf")
                except Exception as e:
                    st.error(f"Eroare la rularea ARIMA: {e}")

    elif current_page == "Predictie RNN":
        st.subheader("📉 Predicție RNN pentru curs valutar")

        uploaded_file = st.file_uploader("Încarcă fișierul CSV cu date valutare", type="csv", key="rnn_csv")

        if uploaded_file is not None:
            with open("currency_prediction_data_set.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())

            df = pd.read_csv("currency_prediction_data_set.csv", index_col=0)
            df.index = pd.to_datetime(df.index)
            st.write("Date încărcate:", df.head())

            valuta_selectata = st.selectbox("Alege valuta pentru predicție:", [col[4:] for col in df.columns], key="valuta_rnn")

            predict_type = st.radio("Tipul de predicție:", ["Pentru mai multe zile", "Până la o dată aleasă"])

            if predict_type == "Pentru mai multe zile":
                forecast_days = st.slider("Număr de zile de prezis:", min_value=1, max_value=30, value=10, key="forecast_days_rnn")
            else:
                ultima_zi = pd.to_datetime(df.index[-1])
                data_selectata = st.date_input("Alege data țintă pentru predicție:", min_value=ultima_zi + timedelta(days=1))
                forecast_days = (data_selectata - ultima_zi.date()).days

            if st.button("Execută predicția RNN"):
                try:
                    result = rnn_model(valuta_selectata, forecast_days)

                    st.write("📊 Rezultatele predicției RNN:")
                    forecast_df = pd.DataFrame({
                        "Ziua": [f"Ziua {i+1}" for i in range(forecast_days)],
                        "Valoare prezisă": result["future_predictions"]
                    })
                    st.dataframe(forecast_df)

                    with open("predictie_rnn.pdf", "rb") as f:
                        st.download_button("📥 Descarcă graficul PDF", data=f, file_name="predictie_rnn.pdf")
                except Exception as e:
                    st.error(f"Eroare la rularea modelului RNN: {e}")

    elif current_page == "Delogare":
        st.info("Ești pe cale să te deloghezi.")
        if st.button("Confirmă Delogarea", key="confirm_logout_btn_app"):
            st.session_state.clear()
            st.rerun()
