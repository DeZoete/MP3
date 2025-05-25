import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def show_map_institution():
    st.header("📍 Kortvisning: Frafald og fuldførelse pr. institution")

    # 1. Læs originalfilen direkte
    file_path = "Streamlit/Data/Afbrudte_og_fuldførte_institution.xlsx"
    df = pd.read_excel(file_path)

    # 2. Gruppér efter Subinstitution og summér
    grouped = df.groupby("Subinstitution")[["Afbrudte", "Fuldførte"]].sum().reset_index()

    # 3. Tilføj koordinater via mapping
    coordinates_map = {
        "Københavns Professionshøjskole": (55.6909, 12.5529),
        "Professionshøjskolen VIA University College": (56.1629, 10.2039),
        "Erhvervsakademi Aarhus": (56.1629, 10.2039),
        "Professionshøjskolen University College Nordjylland": (57.0488, 9.9217),
        "Erhvervsakademiet Copenhagen Business Academy": (55.6759, 12.5655),
        "University College Lillebælt": (55.4038, 10.4024),
        "University College Sjælland": (55.4377, 11.5666),
        "University College Syddanmark": (55.4904, 9.4722),
        "Erhvervsakademi Dania": (56.4604, 10.0364),
        "Erhvervsakademi SydVest": (55.4765, 8.4594),
        "Erhvervsakademi MidtVest": (56.3615, 8.6164),
        "Erhvervsakademi Sjælland": (55.4580, 11.5820),
        "IBA Erhvervsakademi Kolding": (55.4910, 9.4720),
        "Erhvervsakademi Bornholm": (55.1037, 14.7065),
        "Erhvervsakademi Nordjylland": (57.0488, 9.9217),
    }

    grouped["lat"] = grouped["Subinstitution"].map(lambda x: coordinates_map.get(x, (None, None))[0])
    grouped["lon"] = grouped["Subinstitution"].map(lambda x: coordinates_map.get(x, (None, None))[1])

    grouped = grouped.dropna(subset=["lat", "lon"])

    if grouped.empty:
        st.warning("Ingen koordinater matchede institutionerne. Tilføj flere til mapping-tabellen.")
        return

    # 4. Vis kortet
    fig = px.scatter_mapbox(
        grouped,
        lat="lat",
        lon="lon",
        size="Afbrudte",
        color="Fuldførte",
        hover_name="Subinstitution",
        hover_data=["Afbrudte", "Fuldførte"],
        color_continuous_scale="Viridis",
        size_max=25,
        zoom=6,
        mapbox_style="open-street-map",
        title="Frafald og fuldførelse fordelt på institution"
    )

    st.plotly_chart(fig, use_container_width=True)

def histogram(data, column_name, title="Histogram"):
    fig, ax = plt.subplots()
    ax.hist(data[column_name], bins=10, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel(column_name)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def boxplot(data, column_name, title="Box Plot"):
    fig, ax = plt.subplots()
    sns.boxplot(y=data[column_name], ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def scatter_plot(data, x_column, y_column, title="Scatter Plot"):
    fig, ax = plt.subplots()
    ax.scatter(data[x_column], data[y_column])
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    st.pyplot(fig)

def scatter_plot_3d(data, x_column, y_column, z_column, title="3D Scatter Plot", color_column=None):
    st.subheader(title)
    fig = px.scatter_3d(data, x=x_column, y=y_column, z=z_column, color=color_column, opacity=0.7)
    fig.update_layout(title=title)
    st.plotly_chart(fig)
    
    


def show_graphsInstitutioner():
    st.header("Visualisering af data")

    # Læs data
    data = pd.read_excel("Streamlit/Data/Afbrudte_og_fuldførte_institution.xlsx")

    # Rens data (fjerner 'Hovedinstitution' m.m.)
    data = data[~data["Institution"].isin(["Institution", "HovedInstitutionTx", "Hovedinstitution"])]
    data = data.sort_values("Fuldførte", ascending=False)
    

   
    st.subheader("Antal fuldførte pr. institution")
    fig1 = px.bar(
        data,
        x="InstitutionType",
        y="Fuldførte",
        title="Fuldførte pr. InstitutionType",
        hover_name="InstitutionType"
    )
    fig1.update_layout(xaxis={'visible': False}, width=1200, height=500)  # Skjul labels, bred grafik
    st.plotly_chart(fig1)

    data_sorted_afbrudte = data.sort_values("Afbrudte", ascending=False)
    st.subheader("Antal afbrudte pr. institution")
    fig2 = px.bar(
        data_sorted_afbrudte,
        x="InstitutionType",
        y="Afbrudte",
        title="Afbrudte pr. InstitutionType",
        hover_name="InstitutionType"
    )
    fig2.update_layout(xaxis={'visible': False}, width=1200, height=500)
    st.plotly_chart(fig2)




def show_graphsInstitutionerSelvValgt():
    st.title("Analyse af Frafald pr. Institution")

    # Læs og rens data
    data = pd.read_excel("Streamlit/Data/Afbrudte_og_fuldførte_institution.xlsx")
    data = data[~data["Institution"].isin(["Institution", "HovedInstitutionTx", "Hovedinstitution"])]
    # Fylder NaN-værdier med 0
    data[["Afbrudte", "Fuldførte"]] = data[["Afbrudte", "Fuldførte"]].fillna(0)
    # Fjerner rækker hvor både Afbrudte og Fuldførte er 0, da institutioner hvor begge er 0 
    # indikerer, at instituionen ikke er oprettet endnu, eller er blevet nedlagt
    # og derfor ikke er relevant for analysen
    data = data[~((data["Afbrudte"] == 0) & (data["Fuldførte"] == 0))]

    # Beregn frafaldsrate
    data["Frafaldsrate"] = 100 * data["Afbrudte"] / (data["Afbrudte"] + data["Fuldførte"])

    # Bruger vælger institutionstype
    st.header("Vælg institutionstype")
    st.markdown("""
    **Bemærk:**  
    Nogle institutioner har 0 fuldførte, men et højt antal afbrudte.  
    Dette kan skyldes, at institutionen er blevet **nedlagt** i løbet af perioden,  
    og derfor ikke har haft mulighed for at fuldføre forløb.
    
    Hvis en institutionstype har få afbrudte og eller fuldførte og ingen fuldførte og eller afbrudte, 
    skyldes det, at der har været mindre end 5 studerende der har fuldført eller afbrudt.
    Dette er for at beskytte anonymiteten af de studerende.
    """)

    valgte_institutionstyper = sorted(data["InstitutionType"].unique())
    valgt_insttype = st.selectbox("Vælg institutionstype", valgte_institutionstyper)

    # Filtrér efter valgt institutionstype
    inst_data = data[data["InstitutionType"] == valgt_insttype]

    # Vælg år eller 'Alle år'
    mulige_år = sorted(inst_data["År"].dropna().unique())
    valgt_år = st.selectbox("Vælg år (eller se alle)", ["Alle år"] + list(map(str, mulige_år)))

    if valgt_år != "Alle år":
        inst_data = inst_data[inst_data["År"] == int(valgt_år)]

    if inst_data.empty:
        st.warning("Ingen data fundet for det valgte valg.")
        return

    # Aggregér hvis alle år
    if valgt_år == "Alle år":
        samlet = inst_data[["Afbrudte", "Fuldførte"]].sum()
        frafald = 100 * samlet["Afbrudte"] / (samlet["Afbrudte"] + samlet["Fuldførte"])
    else:
        samlet = inst_data.iloc[0]
        frafald = samlet["Frafaldsrate"]

    st.subheader(f"Statistik for: {valgt_insttype} ({valgt_år})")
    st.metric("Fuldførte", int(samlet["Fuldførte"]))
    st.metric("Afbrudte", int(samlet["Afbrudte"]))
    st.metric("Frafaldsrate (%)", round(frafald, 2))

    st.divider()

    # Bar chart: Fuldførte og afbrudte
    st.subheader("Sammenligning af Fuldførte og Afbrudte")
    if valgt_år == "Alle år":
        bar_data = pd.DataFrame({
            "Status": ["Fuldførte", "Afbrudte"],
            "Antal": [samlet["Fuldførte"], samlet["Afbrudte"]]
        })
    else:
        bar_data = inst_data.melt(id_vars=["InstitutionType", "År"], value_vars=["Fuldførte", "Afbrudte"],
                                  var_name="Status", value_name="Antal")

    fig = px.bar(bar_data, x="Status", y="Antal", title=f"Fuldførte vs. Afbrudte ({valgt_insttype} - {valgt_år})")
    st.plotly_chart(fig)

    st.divider()

    # Sammenlign frafaldsrate med andre institutionstyper (kun hvis 'Alle år' er valgt)
    if valgt_år == "Alle år":
        st.subheader("Sammenlign frafaldsrate med andre institutionstyper")
        fig2 = px.box(data, x="InstitutionType", y="Frafaldsrate", title="Frafaldsrate fordelt på InstitutionType", points="all")
        st.plotly_chart(fig2)





# VISUALISERING: Faglinje og grafer

# Funktion: Histogram
def histogram(data, column_name, title="Histogram"):
    fig, ax = plt.subplots()
    ax.hist(data[column_name], bins=10, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel(column_name)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# Funktion: Boxplot
def boxplot(data, column_name, title="Box Plot"):
    fig, ax = plt.subplots()
    sns.boxplot(y=data[column_name], ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

# Funktion: Scatter plot (3D)
def scatter_plot_3d(data, x_column, y_column, z_column, title="3D Scatter Plot", color_column=None):
    st.subheader(title)
    fig = px.scatter_3d(
        data,
        x=x_column,
        y=y_column,
        z=z_column,
        color=color_column,
        opacity=0.7
    )
    fig.update_layout(title=title)
    st.plotly_chart(fig)

# Funktion: Hovedvisualisering
def show_graphs():
    st.title("Analyse af Frafald på Videregående Uddannelser")

    # Indlæs data
    file_path = "Streamlit/Data/Uddannelse_combined.xlsx"
    df = pd.read_excel(file_path)
    years = list(range(2015, 2025))

    # Split data
    fuldfort = df[df['Type'] == 'Fuldført']
    afbrudt = df[df['Type'] == 'Afbrudt']

    # Aggreger på FagLinje og FagRetning
    agg_fuldfort = fuldfort.groupby(['FagLinjer','FagRetning'])[years].sum().reset_index()
    agg_afbrudt = afbrudt.groupby(['FagLinjer','FagRetning'])[years].sum().reset_index()

    # Merge og beregn frafaldsrate
    ret_merged = pd.merge(agg_fuldfort, agg_afbrudt, on=['FagLinjer','FagRetning'], suffixes=('_fuldfort', '_afbrudt'))
    ret_merged['Total_fuldfort'] = ret_merged[[f"{y}_fuldfort" for y in years]].sum(axis=1)
    ret_merged['Total_afbrudt'] = ret_merged[[f"{y}_afbrudt" for y in years]].sum(axis=1)
    ret_merged['Frafaldsrate'] = 100 * ret_merged['Total_afbrudt'] / (ret_merged['Total_fuldfort'] + ret_merged['Total_afbrudt'])

    # Første valg: FagLinje
    st.header("Trin 1: Vælg en FagLinje")
    alle_linjer = sorted(ret_merged['FagLinjer'].unique())
    valgt_linje = st.selectbox("Vælg en FagLinje", alle_linjer)

    linje_data = ret_merged[ret_merged['FagLinjer'] == valgt_linje]

    if linje_data.empty:
        st.warning("Ingen data fundet for den valgte FagLinje.")
        return

    # Tilføj bar chart (Fuldført vs. Afbrudt for valgt FagLinje)
    grouped_linje = df[df['FagLinjer'] == valgt_linje].groupby("Type")[years].sum().transpose()
    st.subheader(f"Fuldført vs. Afbrudt for: {valgt_linje}")
    st.bar_chart(grouped_linje)
    st.dataframe(grouped_linje)

    st.divider()

    # Vis FagRetninger med frafaldsrate
    st.subheader(f"Frafaldsrate for FagRetninger under {valgt_linje}")
    st.dataframe(linje_data[['FagRetning', 'Frafaldsrate']].sort_values(by="Frafaldsrate", ascending=False))

    # Andet valg: FagRetning
    st.header("Trin 2: Vælg en FagRetning under den valgte FagLinje")
    retninger = linje_data['FagRetning'].unique()
    valgt_retning = st.selectbox("Vælg en FagRetning", retninger)

    valgte_data = linje_data[linje_data['FagRetning'] == valgt_retning]

    if valgte_data.empty:
        st.warning("Ingen data for den valgte kombination.")
        return

    row = valgte_data.iloc[0]
    fuldførte = row[[f"{y}_fuldfort" for y in years]].values
    afbrudte = row[[f"{y}_afbrudt" for y in years]].values
    total = fuldførte + afbrudte
    frafaldsrate = 100 * afbrudte / total

    st.subheader(f"Tidsserie for {valgt_retning} under {valgt_linje}")
    st.line_chart(pd.DataFrame({
        "Fuldført": fuldførte,
        "Afbrudt": afbrudte
    }, index=years))

    st.line_chart(pd.DataFrame({
        "Frafaldsrate (%)": frafaldsrate
    }, index=years))
    
    # Ekstra plots
    st.subheader("Fordelingsanalyser")
    if 'Type' in df.columns:
        histogram(df, 'Type', title="Antallet af afbrudte og fuldførte")

    if '2015' in df.columns:
        boxplot(df, '2015', title="Niveau i 2015")
    if '2020' in df.columns:
        boxplot(df, '2020', title="Niveau i 2020")
    if '2023' in df.columns:
        boxplot(df, '2023', title="Niveau i 2023")

    if all(col in df.columns for col in ['2015', '2020', '2023', 'Type']):
        scatter_plot_3d(df, '2015', '2020', '2023', title="Tendenser 2015–2023", color_column='Type')

def show_prediction_model():
    st.header("Forudsig frafald og fuldførelse i 2025 med lineær regression")

    # Indlæs data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = os.path.join(base_dir, 'Streamlit', 'Data', 'Uddannelse_combined.xlsx')
    df = pd.read_excel(excel_path)
    df.columns = df.columns.map(str)

    all_years = [str(y) for y in range(2015, 2025)]
    df[all_years] = df[all_years].apply(pd.to_numeric, errors='coerce').fillna(0)

    df_afbrudt = df[df["Type"] == "Afbrudt"].copy().reset_index(drop=True)
    df_fuldført = df[df["Type"] == "Fuldført"].copy().reset_index(drop=True)

    # MODEL 1: Afbrudt
    train_features = [str(y) for y in range(2015, 2024)]
    X_ab_train = df_afbrudt[train_features]
    y_ab_train = df_afbrudt["2024"]
    model_ab = LinearRegression()
    model_ab.fit(X_ab_train, y_ab_train)
    X_ab_2025 = df_afbrudt[[str(y) for y in range(2016, 2025)]].copy()
    X_ab_2025.columns = train_features
    y_pred_ab_2025 = model_ab.predict(X_ab_2025)

    # MODEL 2: Fuldført
    X_fu_train = df_fuldført[train_features]
    y_fu_train = df_fuldført["2024"]
    model_fu = LinearRegression()
    model_fu.fit(X_fu_train, y_fu_train)
    X_fu_2025 = df_fuldført[[str(y) for y in range(2016, 2025)]].copy()
    X_fu_2025.columns = train_features
    y_pred_fu_2025 = model_fu.predict(X_fu_2025)

    # VISNING
    df_vis = df_afbrudt[["Uddannelse", "FagLinjer", "FagRetning", "2024"]].copy()
    df_vis.rename(columns={"2024": "2024_afbrudt"}, inplace=True)
    df_vis["2025_afbrudt (forudsagt)"] = y_pred_ab_2025

    df_fu = df_fuldført[["Uddannelse", "FagLinjer", "FagRetning", "2024"]].copy()
    df_fu.rename(columns={"2024": "2024_fuldført"}, inplace=True)
    df_fu["2025_fuldført (forudsagt)"] = y_pred_fu_2025

    df_vis = pd.merge(df_vis, df_fu, on=["Uddannelse", "FagLinjer", "FagRetning"], how="outer")

    df_vis["Frafaldsprocent_2025"] = df_vis["2025_afbrudt (forudsagt)"] / (
        df_vis["2025_afbrudt (forudsagt)"] + df_vis["2025_fuldført (forudsagt)"]
    ) * 100

    st.subheader("Tabel med forudsagte værdier for 2025")
    st.dataframe(df_vis)

    st.subheader("Top 20 fagretninger – forudsagt frafald i 2025")
    st.bar_chart(df_vis.sort_values("2025_afbrudt (forudsagt)", ascending=False).head(20).set_index("FagRetning")["2025_afbrudt (forudsagt)"])

    st.subheader("Top 20 fagretninger – forudsagt fuldført i 2025")
    st.bar_chart(df_vis.sort_values("2025_fuldført (forudsagt)", ascending=False).head(20).set_index("FagRetning")["2025_fuldført (forudsagt)"])

    st.subheader("Top 20 fagretninger – forudsagt frafaldsprocent i 2025")
    st.bar_chart(df_vis.sort_values("Frafaldsprocent_2025", ascending=False).head(20).set_index("FagRetning")["Frafaldsprocent_2025"])

    # VISUALISERING af historik og forudsigelse
    st.subheader("Visualisering af regression for valgt fagretning (separat for afbrudt og fuldført)")
    fagretninger = df_vis["FagRetning"].dropna().unique()
    valgt_fagretning = st.selectbox("Vælg en fagretning", fagretninger)

    # Filtrér data for valgt fagretning
    row_ab = df_afbrudt[df_afbrudt["FagRetning"] == valgt_fagretning].reset_index(drop=True)
    row_fu = df_fuldført[df_fuldført["FagRetning"] == valgt_fagretning].reset_index(drop=True)

    if row_ab.empty or row_fu.empty:
        st.warning("Valgt fagretning findes ikke i både afbrudt og fuldført data.")
        return

    idx_ab = row_ab.index[0]
    idx_fu = row_fu.index[0]

    år = list(range(2015, 2025))

    # --------- Plot 1: Afbrudt ---------
    y_ab = row_ab.iloc[0][[str(y) for y in år]].values
    y_2025_ab = y_pred_ab_2025[idx_ab]

    fig_ab, ax_ab = plt.subplots()
    ax_ab.plot(år, y_ab, marker='o', label="Afbrudt 2015–2024")
    ax_ab.plot(2025, y_2025_ab, 'go', label="Afbrudt 2025 (forudsagt)")
    ax_ab.set_title(f"Afbrudt – {valgt_fagretning}")
    ax_ab.set_xlabel("År")
    ax_ab.set_ylabel("Antal studerende")
    ax_ab.legend()
    st.pyplot(fig_ab)

    # --------- Plot 2: Fuldført ---------
    y_fu = row_fu.iloc[0][[str(y) for y in år]].values
    y_2025_fu = y_pred_fu_2025[idx_fu]

    fig_fu, ax_fu = plt.subplots()
    ax_fu.plot(år, y_fu, marker='x', linestyle='--', label="Fuldført 2015–2024")
    ax_fu.plot(2025, y_2025_fu, 'ro', label="Fuldført 2025 (forudsagt)")
    ax_fu.set_title(f"Fuldført – {valgt_fagretning}")
    ax_fu.set_xlabel("År")
    ax_fu.set_ylabel("Antal studerende")
    ax_fu.legend()
    st.pyplot(fig_fu)