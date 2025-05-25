import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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





def show_institution_clustering():
    st.header("Clustering af institutioner baseret på frafald og fuldførelse")
    df = pd.read_excel("Streamlit/Data/Afbrudte_og_fuldførte_institution.xlsx")
    df = df[~df["Institution"].isin(["Institution", "HovedInstitutionTx", "Hovedinstitution"])]
    df[["Afbrudte", "Fuldførte"]] = df[["Afbrudte", "Fuldførte"]].fillna(0)
    df = df[~((df["Afbrudte"] == 0) & (df["Fuldførte"] == 0))]
    df["dropout_rate"] = df["Afbrudte"] / (df["Afbrudte"] + df["Fuldførte"])

    # Select features for clustering
    features = df[["Afbrudte", "Fuldførte", "dropout_rate"]]
    features = features.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    st.write("Cluster centers (scaled):", kmeans.cluster_centers_)

    # Visualize clusters
    fig = px.scatter(df, x="Afbrudte", y="Fuldførte", color="Cluster",
                     hover_data=["Subinstitution", "dropout_rate"],
                     title="Institutioner grupperet efter frafald og fuldførelse")
    st.plotly_chart(fig)




def show_feature_importance():
    st.header("Feature importance for frafaldsrate (inkl. region)")

    # Indlæs data
    df = pd.read_excel("Streamlit/Data/Afbrudte_og_fuldførte_institution.xlsx")
    df = df[~df["Institution"].isin(["Institution", "HovedInstitutionTx", "Hovedinstitution"])]
    df[["Afbrudte", "Fuldførte"]] = df[["Afbrudte", "Fuldførte"]].fillna(0)
    df = df[~((df["Afbrudte"] == 0) & (df["Fuldførte"] == 0))]
    df["dropout_rate"] = df["Afbrudte"] / (df["Afbrudte"] + df["Fuldførte"])

    # Tilføj region
    region_map = {
        "Københavns Professionshøjskole": "Sjælland",
        "Professionshøjskolen VIA University College": "Jylland",
        "Erhvervsakademi Aarhus": "Jylland",
        "Professionshøjskolen University College Nordjylland": "Jylland",
        "Erhvervsakademiet Copenhagen Business Academy": "Sjælland",
        "University College Lillebælt": "Fyn",
        "University College Sjælland": "Sjælland",
        "University College Syddanmark": "Jylland",
        "Erhvervsakademi Dania": "Jylland",
        "Erhvervsakademi SydVest": "Jylland",
        "Erhvervsakademi MidtVest": "Jylland",
        "Erhvervsakademi Sjælland": "Sjælland",
        "IBA Erhvervsakademi Kolding": "Jylland",
        "Erhvervsakademi Bornholm": "Bornholm",
        "Erhvervsakademi Nordjylland": "Jylland",
    }
    df["Region"] = df["Subinstitution"].map(region_map)
    df = df.dropna(subset=["Region", "dropout_rate"])

    # Encode categorical features (Region, År, InstitutionType)
    X = pd.get_dummies(df[["År", "InstitutionType", "Region"]])
    y = df["dropout_rate"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances_percent = importances * 100  # Show as percent

    # Use Plotly for a better bar chart
    importances_df = importances_percent.sort_values(ascending=False).reset_index()
    importances_df.columns = ["Feature", "Importance (%)"]
    fig = px.bar(importances_df, x="Feature", y="Importance (%)", title="Feature Importance for Dropout Rate")
    st.plotly_chart(fig)

    st.write("De vigtigste features for at forudsige frafaldsrate (%):", importances_df.head(10))

    # Show predictions vs. actual
    df["Predicted_dropout_rate"] = y_pred
    st.subheader("Faktisk vs. forudsagt frafaldsrate")
    st.dataframe(df[["År", "InstitutionType", "Region", "dropout_rate", "Predicted_dropout_rate"]])

    # Show R2 score
    from sklearn.metrics import r2_score
    r2 = r2_score(y, y_pred)
    st.write(f"R²-score for modellen: {r2:.3f}")



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
    y_pred_ab_2025 = model_ab.predict(X_ab_2025).round().astype(int)

    # MODEL 2: Fuldført
    X_fu_train = df_fuldført[train_features]
    y_fu_train = df_fuldført["2024"]
    model_fu = LinearRegression()
    model_fu.fit(X_fu_train, y_fu_train)
    X_fu_2025 = df_fuldført[[str(y) for y in range(2016, 2025)]].copy()
    X_fu_2025.columns = train_features
    y_pred_fu_2025 = model_fu.predict(X_fu_2025).round().astype(int)

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



def show_uddannelse_prediction_model():
    st.header("📈 Forudsigelse og Analyse af Frafald")
    
    file_path = "Streamlit/Data/Uddannelse_combined.xlsx"
    df = pd.read_excel(file_path)

    fuldfort = df[df['Type'] == 'Fuldført']
    afbrudt = df[df['Type'] == 'Afbrudt']
    years = list(range(2015, 2025))

    # Aggregering
    agg_fuldfort = fuldfort.groupby(['FagLinjer'])[years].sum().reset_index()
    agg_afbrudt = afbrudt.groupby(['FagLinjer'])[years].sum().reset_index()
    merged = pd.merge(agg_fuldfort, agg_afbrudt, on='FagLinjer', suffixes=('_fuldfort', '_afbrudt'))
    merged['Total_fuldfort'] = merged[[f"{y}_fuldfort" for y in years]].sum(axis=1)
    merged['Total_afbrudt'] = merged[[f"{y}_afbrudt" for y in years]].sum(axis=1)
    merged['Frafaldsrate'] = merged['Total_afbrudt'] / (merged['Total_fuldfort'] + merged['Total_afbrudt'])

    for y in years:
        merged[f'{y}_ratio'] = merged[f'{y}_afbrudt'] / (merged[f'{y}_fuldfort'] + merged[f'{y}_afbrudt'] + 1e-6)

    # Tabs for opdeling
    tabs = st.tabs(["📊 Modellering", "🔍 Klyngeanalyse", "📈 Forudsigelse 2025", "📘 Baggrund"])

    # === MODELTAB ===
    with tabs[0]:
        st.text_input("Beskrivelse af analysen", "Skriv tekst her")

        features = merged[
            [f"{y}_fuldfort" for y in years] +
            [f"{y}_afbrudt" for y in years] +
            [f"{y}_ratio" for y in years]
        ]
        target = merged['Frafaldsrate']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.1),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            st.subheader(f"{name} – R²: {r2:.3f}")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.7)
            ax.plot([0, 1], [0, 1], 'r--')
            ax.set_xlabel("True Frafaldsrate")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

    # === KLYNGETAB ===
    with tabs[1]:
        st.text_input("Beskrivelse af klyngeanalyse", "Skriv tekst her")
        st.subheader("PCA + DBSCAN Klyngeanalyse")

        X_cluster = merged[['Frafaldsrate']]
        X_scaled = StandardScaler().fit_transform(X_cluster)
        db = DBSCAN(eps=0.5, min_samples=2)
        labels = db.fit_predict(X_scaled)

        X_pca = StandardScaler().fit_transform(
            merged[[f"{y}_fuldfort" for y in years] + [f"{y}_afbrudt" for y in years]]
        )
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_pca)

        fig, ax = plt.subplots(figsize=(10, 6))
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(components[mask, 0], components[mask, 1], label=f"Klynge {label}", color=color)
            for i in np.where(mask)[0]:
                ax.text(components[i, 0], components[i, 1], merged.iloc[i]['FagLinjer'], fontsize=8)
        ax.set_title("PCA + DBSCAN Klyngeanalyse")
        ax.set_xlabel("Komponent 1")
        ax.set_ylabel("Komponent 2")
        ax.legend()
        st.pyplot(fig)

    # === FORUDSIGELSE 2025 ===
    with tabs[2]:
        st.text_input("Forudsigelse for 2025", "Skriv tekst her")
        st.subheader("Forudsigelse 2025 per faglinje")

        pivot_df = afbrudt.pivot_table(index='FagLinjer', values=years, aggfunc='sum').fillna(0)
        X = np.array(years).reshape(-1, 1)
        faglinje_predictions = {}
        for fag, row in pivot_df.iterrows():
            y = row.values
            lr = LinearRegression().fit(X, y)
            rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
            faglinje_predictions[fag] = {
                'LR_2025': lr.predict([[2025]])[0],
                'RF_2025': rf.predict([[2025]])[0],
                'LR_R2': lr.score(X, y),
                'RF_R2': r2_score(y, rf.predict(X))
            }

        pred_df = pd.DataFrame(faglinje_predictions).T
        pred_df['Forskel_2025'] = pred_df['RF_2025'] - pred_df['LR_2025']
        pred_df['Forskel_pct'] = (pred_df['Forskel_2025'] / pred_df['LR_2025']) * 100
        st.dataframe(pred_df.sort_values(by='RF_2025', ascending=False).round(2).head(10))

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sorted_df = pred_df.sort_values(by='RF_2025', ascending=False)
        ax2.bar(range(len(sorted_df.index[:10])), sorted_df['RF_2025'].head(10), color='skyblue')
        ax2.set_title("Top 10: RF-forudsagt frafald i 2025")
        ax2.set_ylabel("Antal studerende")
        ax2.set_xticks(range(len(sorted_df.index[:10])))
        ax2.set_xticklabels(sorted_df.index[:10], rotation=45, ha='right')
        st.pyplot(fig2)

    # === BAGGRUND ===
    with tabs[3]:
        st.markdown("""\
### Baggrund: Frafald blandt pædagogstuderende

Regeringen har foreslået en ny erhvervsrettet ungdomsuddannelse, EPX, der skal give en mere direkte vej til pædagoguddannelsen. Formålet er at reducere frafaldet ved at tilbyde en praksisnær tilgang. Dog udtrykker fagfolk bekymring for, at en kortere uddannelsesvej kan føre til mindre modne og fagligt forberedte studerende, hvilket potentielt kan øge frafaldet yderligere.

[Kilde: Politiken – Flere pædagogstuderende dropper ud – nu reagerer ministeren](https://politiken.dk/danmark/art9814464/Flere-p%C3%A6dagogstuderende-dropper-ud-%E2%80%93-nu-reagerer-ministeren)
""")

        st.markdown("""\
### 📘 Resumé: Frafald på sundhedsuddannelserne ved VIA University College

Analysen fra Danmarks Evalueringsinstitut (EVA) identificerer flere centrale årsager til, at studerende på sundhedsuddannelserne ved VIA University College vælger at afbryde deres uddannelse:

- **Udfordringer med faglig og social integration**: Mange studerende oplever vanskeligheder med at tilpasse sig det akademiske niveau og opbygge sociale relationer.
- **Manglende forberedelse fra tidligere uddannelser**: Studerende fra fx HF eller med lave karakterer fra folkeskolen har højere frafald.
- **Personlige og økonomiske forhold**: Helbred, økonomi og familieforhold påvirker studiegennemførelse.
- **Manglende støtte og vejledning**: Begrænset adgang til vejledning forværrer problemer.

Kilde: Danmarks Evalueringsinstitut (EVA), *Analyse af frafald på VIA University College – Sundhed*, 2016.  
[Se hele rapporten her](https://eva.dk/Media/638409044635990892/Analyse%20af%20frafald%20p%C3%A5%20VIA%20University%20College%20-%20Sundhed.pdf)
""")
