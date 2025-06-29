import streamlit as st
import yaml
import bcrypt
import os
from PIL import Image

# Load user data
def load_users():
    with open("users.yaml", "r") as f:
        return yaml.safe_load(f)

def save_users(data):
    with open("users.yaml", "w") as f:
        yaml.dump(data, f)

if "logged_in" in st.session_state and st.session_state["logged_in"]:
    if st.session_state.get("current_page") == "Setări":
        st.subheader("⚙️ Setări cont")

        users_data = load_users()
        username = st.session_state["username"]
        user_info = users_data["credentials"]["usernames"][username]

        
        st.markdown(f"**Email curent:** {user_info['email']}")

        
        profile_path = user_info.get("profile_image")
        if profile_path and os.path.exists(profile_path):
            st.image(profile_path, caption="Poza de profil actuală", width=100)

        
        with st.expander("🔑 Schimbă parola"):
            new_password = st.text_input("Parolă nouă", type="password", key="new_password")
            confirm_password = st.text_input("Confirmă parola", type="password", key="confirm_password")

            if st.button("Actualizează parola", key="btn_update_password"):
                if new_password and new_password == confirm_password:
                    hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                    users_data["credentials"]["usernames"][username]["password"] = hashed_pw
                    save_users(users_data)
                    st.success("Parola a fost actualizată cu succes!")
                else:
                    st.error("Parolele nu se potrivesc sau sunt goale.")

        
        with st.expander("📧 Schimbă emailul"):
            new_email = st.text_input("Email nou", key="new_email")
            if st.button("Actualizează email", key="btn_update_email"):
                if new_email:
                    users_data["credentials"]["usernames"][username]["email"] = new_email
                    save_users(users_data)
                    st.success("Emailul a fost actualizat cu succes!")
                else:
                    st.error("Emailul nu poate fi gol.")

       
        with st.expander("🖼️ Încarcă o imagine de profil"):
            uploaded_file = st.file_uploader("Alege o imagine", type=["jpg", "jpeg", "png"], key="file_uploader")
            if uploaded_file is not None:
               
                st.image(uploaded_file, caption="Previzualizare imagine", width=150)

                if st.button("Salvează imaginea de profil", key="btn_save_image"):
                    os.makedirs("profile_images", exist_ok=True)
                    ext = uploaded_file.name.split(".")[-1]
                    image_path = os.path.join("profile_images", f"{username}.{ext}")
                    with open(image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    users_data["credentials"]["usernames"][username]["profile_image"] = image_path
                    save_users(users_data)
                    st.success("Imaginea a fost salvată cu succes!")

       
        with st.expander("🔔 Notificări"):
            current_pref = user_info.get("notifications", False)
            notify = st.checkbox("Activează notificări prin email", value=current_pref, key="notif_checkbox")
            if st.button("Salvează preferințele", key="btn_save_notifications"):
                users_data["credentials"]["usernames"][username]["notifications"] = notify
                save_users(users_data)
                st.success("Preferințele pentru notificări au fost salvate.")
else:
    st.warning("Trebuie să fii autentificat pentru a accesa setările.")
