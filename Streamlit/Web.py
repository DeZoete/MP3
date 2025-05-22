import streamlit as st
import sys
import os

# Sørg for at vi kan importere fra stats-mappen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importer statistikmoduler
from stats import correlation
from stats import dispersion
from stats import central_tendency
from stats import visualization

# Forside-indhold
def show_homepage():
    st.header("📊 Uddannelse Data Analysis")
    st.write("Velkommen til vores BI-analyseværktøj for uddannelse og frafald.")
    st.write("Brug menuen til venstre for at se grafer eller prøve en forudsigelsesmodel.")

# Hovedfunktion
def main():
    # Skal være allerførst!
    st.set_page_config(page_title="Uddannelse BI", layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Vælg en side", ["Homepage", "Visualization", "Prediction"])

    try:
        if page == "Homepage":
            show_homepage()
        elif page == "Visualization":
            visualization.show_graphs()
        elif page == "Prediction":
            visualization.show_prediction_model()
    except Exception as e:
        st.error(f"⚠️ Fejl under visning af siden: {e}")

# Kør kun hvis scriptet startes direkte
if __name__ == "__main__":
    main()
