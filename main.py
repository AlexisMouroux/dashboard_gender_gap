import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
import prince
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
import streamlit.components.v1 as components




# Charger les données
monmaster2024 = pd.read_csv('monmaster_data_info_2024.csv')
monmaster2023 = pd.read_csv('monmaster_data_info_2023.csv')

parcoursup2024 = pd.read_csv('parcoursup_data_info_2024.csv')
parcoursup2023 = pd.read_csv('parcoursup_data_info_2023.csv')
parcoursup2022 = pd.read_csv('parcoursup_data_info_2022.csv')
parcoursup2021 = pd.read_csv('parcoursup_data_info_2021.csv')
parcoursup2020 = pd.read_csv('parcoursup_data_info_2020.csv')
parcoursup2019 = pd.read_csv('parcoursup_data_info_2019.csv')

# Ajout de la colonne Année et Niveau pour les datasets Parcoursup
for year, df in zip(range(2019, 2025), [parcoursup2019, parcoursup2020, parcoursup2021, parcoursup2022, parcoursup2023, parcoursup2024]):
    df['Année'] = year
    df['Niveau'] = 'Bac +2/3'


# Ajout de la colonne Année et Niveau pour les datasets MonMaster
for year, df in zip(range(2023, 2025), [monmaster2023, monmaster2024]):
    df['Année'] = year
    df['Niveau'] = 'Bac +5'

# Fusion des datasets parcoursup
parcoursup_data_merged = pd.concat([parcoursup2019, parcoursup2020, parcoursup2021, parcoursup2022, parcoursup2023, parcoursup2024], ignore_index=True)

# Fusion des datasets monmaster
monmaster_data_merged = pd.concat([monmaster2023, monmaster2024], ignore_index=True)




# Fonction pour mapper chaque institution à sa catégorie respective pour Parcoursup
def map_type_filiere_parcoursup(filiere, year):
    if pd.isna(filiere) or not isinstance(filiere, str):  # Vérifie si NaN ou autre type
        return "Inconnu"
    
    filiere_lower = filiere.strip().lower()
    if year == 2020 and "dut" in filiere_lower:
        return "DUT"
    elif year == 2020 and "but" in filiere_lower:
        return "DUT"
    elif "but" in filiere_lower:
        return "BUT"
    elif "bts" in filiere_lower:
        return "BTS"
    elif "licence" in filiere_lower:
        return "Licence"
    elif any(term in filiere_lower for term in ["classe préparatoire", "cycle préparatoire", "cpge", "cupge", "formation d'ingénieur"]):
        return "CPGE"
    else:
        return "Autres"
    

# Fonction pour mapper chaque institution à sa catégorie respective pour MonMaster
def map_type_filiere_monmaster(etablissement):
    if pd.isna(etablissement) or not isinstance(etablissement, str):
        return "Inconnu"
    
    etablissement_lower = etablissement.strip().lower()

    if any(term in etablissement_lower for term in ["faculté","université","recherche"]):
        return "Université"
    elif any(term in etablissement_lower for term in ["formation d'ingénieur","école d'ingénieur","école publique","école nationale"]):
        return "École d'ingénieur"
    elif any(term in etablissement_lower for term in ["école privée","institut"]):
        return "École privée"
    else:
        return "Autres"



   
# Appliquer la fonction de mapping pour toutes les années
monmaster_data_merged['Filière'] = monmaster_data_merged.apply(lambda row: map_type_filiere_monmaster(row['Établissement']), axis=1)
    
# Appliquer la fonction de mapping pour toutes les années
parcoursup_data_merged['Filière'] = parcoursup_data_merged.apply(lambda row: map_type_filiere_parcoursup(row['Filière de formation'], row['Année']), axis=1)


parcoursup_data_merged['Académie de l’établissement'] = parcoursup_data_merged['Académie de l’établissement'].replace({
    'Besançon': 'Besancon',
    'Réunion': 'La Réunion'
})




# Créer une nouvelle colonne "Dont effectif des candidats garçons pour une formation"
monmaster_data_merged['Dont effectif des candidats garçons pour une formation'] = monmaster_data_merged['Effectif total des candidats pour une formation'] - monmaster_data_merged['Dont effectif des candidates pour une formation']




def plot_taux_moyen_admission_filles_academie(df,titre,c1="Académie de l’établissement",c2="Année",c3="% d’admis dont filles"):

    df[c2] = df[c2].astype(float).astype(int).astype(str)


    # Calcul du taux moyen d'admission des filles par académie et année
    taux_par_academie_annee = df.groupby([c1, c2])[c3].mean().reset_index()

    # Création du graphique
    fig = px.bar(
        taux_par_academie_annee,
        x=c1,
        y=c3,
        color=c2,
        barmode="group",
        title=titre,
        labels={
            c3: "Taux d'admission (%)",
            c1: "Académie"
        },
        height=600,
        width=1000,
        color_discrete_sequence=px.colors.qualitative.Safe
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        legend_title_text='Année',
        margin=dict(t=50, l=50, r=50, b=100)
    )

    return fig



def plot_effectif_admises_academie(df,titre,c1="Académie de l’établissement",c2="Année",c3="Dont effectif des candidates admises"):
    # Effectif des admises par académie et par année
    effectif_par_academie_annee = df.groupby([c1, c2])[c3].sum().reset_index()

    # Création du graphique
    fig = px.bar(
        effectif_par_academie_annee,
        x=c1,
        y=c3,
        color=c2,
        barmode="group",
        title=titre,
        labels={
            c3: "Effectif des admises",
            c1: "Académie"
        },
        height=600,
        width=1000,
        color_discrete_sequence=px.colors.qualitative.Safe
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        legend_title_text='Année',
        margin=dict(t=50, l=50, r=50, b=100)
    )

    return fig


def plot_taux_moyen_admission_filles_filiere(df,titre,c1="Filière",c2="% d’admis dont filles"):



    # Calcul du taux moyen d'admission des filles par filière
    taux_par_filiere = df.groupby([c1])[c2].mean().reset_index()

    # Création du graphique
    fig = px.pie(
        taux_par_filiere,
        names=c1,
        values=c2,
        title=titre,
        color_discrete_sequence=px.colors.qualitative.Vivid
    )

    fig.update_traces(textinfo='percent+label')

    fig.update_layout(
        height=600,
        width=800,
        margin=dict(t=50, l=50, r=50, b=50)
    )

    return fig



def plot_taux_admises_par_annee(df,titre,c1="Année",c2="% d’admis dont filles"):
    df[c1] = df[c1].astype(float).astype(int)

    #Taux d'admises par année
    taux_admises = df.groupby(c1)[c2].mean().reset_index()
    taux_admises[c1] = taux_admises[c1].astype(str)

    # Création du graphique
    fig = px.line(
        taux_admises,
        x=c1,
        y=c2,
        markers=True,
        title=titre,
        labels={c2: "Taux moyen (%)"},
        line_shape="linear"
    )
    fig.update_traces(line_color='teal')
    fig.update_layout(height=500, width=800)

    annees = taux_admises[c1].astype(int).tolist()
    fig.update_layout(
        height=500,
        width=800,
        xaxis=dict(
            tickmode='array',
            tickvals=[str(a) for a in annees],
        )
    )

    return fig







# Fusion des datasets Parcoursup et MonMaster
df_merged = pd.concat([monmaster_data_merged, parcoursup_data_merged],ignore_index=True)

#Rennomage des colonnes pour simplifier
df_merged = df_merged.rename(columns={
    "Académie de l’établissement": "Académie",
    "Région de l\'établissement": "Région",
    "Effectif total des candidats pour une formation":"Effectif candidats",
    "Dont effectif des candidats garçons pour une formation":"Effectif candidats garçons",
    "Dont effectif des candidates pour une formation":"Effectif candidates",
})


#Colonnes utilisées
selected_columns = [
    'Filière', 
    'Niveau', 
    'Académie',
    'Région',
    '% admis meme academie',
    'Année',
    '% d’admis dont filles',
    'Effectif candidats',
    'Effectif candidats garçons',
    'Effectif candidates'
    
]


print(df_merged.columns)

# Création d'une copie avec uniquement ces colonnes
df_merged_selection = df_merged[selected_columns].copy()



df_merged_selection = df_merged_selection.dropna(subset=['% admis meme academie'])




#print(df_merged_selection.isna().sum()[df_merged_selection.isna().sum() > 0])




# Sélectionner les colonnes catégorielles
nominales = ['Filière', 'Académie', 'Région']
ordinales = ['Niveau']

for col in nominales:
    df_merged_selection[col] = df_merged_selection[col].astype('category')

# Encoder les colonnes nominales avec OrdinalEncoder
encoder_nominal = OrdinalEncoder()
df_merged_selection[nominales] = encoder_nominal.fit_transform(df_merged_selection[nominales])



# Encoder la colonne ordinale (Niveau) avec un ordre explicite
ordre_niveau = ['Bac +2/3', 'Bac +5']
mapping_ordre = {val: i for i, val in enumerate(ordre_niveau)}
df_merged_selection['Niveau'] = df_merged_selection['Niveau'].map(mapping_ordre)


# Diviser en trois classes basées sur les quantiles
df_merged_selection['classe_filles'] = pd.qcut(
    df_merged_selection["% d’admis dont filles"],
    q=3,
    labels=[0,1,2]
)

# Création d'une copie pour les classes filles
parcoursup_data_classe = parcoursup_data_merged.copy()

parcoursup_data_classe['classe_filles'] = pd.qcut(
    parcoursup_data_classe["% d’admis dont filles"],
    q=3,
    labels=[0,1,2]
)

# Création d'une copie pour les classes filles
monmaster_data_classe = monmaster_data_merged.copy()

monmaster_data_classe['classe_filles'] = pd.qcut(
    monmaster_data_classe["% d’admis dont filles"],
    q=3,
    labels=[0,1,2]
)

def plot_repartition_classes_filles(df, title="Répartition des classes de filles admises par année",x_col="Année", hue_col="classe_filles"):
    classe_labels = {0: "Faible", 1: "Moyen", 2: "Élevé"}
    df["classe_filles"] = df["classe_filles"].map(classe_labels)

    fig = px.histogram(
        df,
        x=x_col,
        color=hue_col,
        barmode="group",  # "stack" si tu préfères les barres empilées
        title=title,
        color_discrete_sequence=px.colors.qualitative.Vivid,
        labels={
            x_col: "Année",
            hue_col: "Quantile % d’admises filles"
        },
        
        category_orders = {"classe_filles": ["Faible", "Moyen", "Élevé"]}
    )

    fig.update_layout(
        legend_title_text="Quantile % d’admises filles",
        xaxis_title="Année",
        yaxis_title="Nombre de formations",
        bargap=0.2
    )

    return fig








colonnes_numeriques = [
    '% admis meme academie',
    '% d’admis dont filles',
    'Effectif candidats',
    'Effectif candidats garçons',
    'Effectif candidates',
]

X = df_merged_selection[colonnes_numeriques]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

def plot_cercle_correlation(pcs, features, titre="Cercle des corrélations (ACP)"):
    x = pcs[0, :]
    y = pcs[1, :]

    fig = go.Figure()

    # Ajouter les vecteurs
    for i in range(len(features)):
        fig.add_trace(go.Scatter(
            x=[0, x[i]],
            y=[0, y[i]],
            mode='lines+text',
            line=dict(color='red'),
            text=[None, features[i]],
            textposition='top center',
            showlegend=False
        ))

    # Ajouter le cercle
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta),
        y=np.sin(theta),
        mode='lines',
        name='Cercle',
        line=dict(color='blue', dash='dash')
    ))

    # Mise en forme
    fig.update_layout(
        title=titre,
        xaxis=dict(title="PC1", range=[-1.1, 1.1], zeroline=True),
        yaxis=dict(title="PC2", range=[-1.1, 1.1], zeroline=True),
        width=700,
        height=700,
        showlegend=False,
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig

# Création d'une copie pour l'acm
df_merged_acm = df_merged.copy()

# Diviser en trois classes basées sur les quantiles
df_merged_acm['classe_filles'] = pd.qcut(
    df_merged_acm["% d’admis dont filles"],
    q=3,
    labels=[0,1,2]
)

colonnes_acm=[
    'Académie',
    'Année',
    'Niveau',
    'Filière',
    'classe_filles'
]
def plot_acm(df, features, titre="Projection des modalités (ACM)"):

    # Préparer les données catégorielles
    X_cat = df[features].astype(str)

    # Appliquer l'ACM
    mca = prince.MCA(n_components=2, random_state=42)
    mca = mca.fit(X_cat)

    # Obtenir les coordonnées des modalités
    coords = mca.column_coordinates(X_cat)
    coords.reset_index(inplace=True)
    coords.columns = ['Modalité', 'Dim 1', 'Dim 2']

    # Extraire la variable d'origine (avant le signe '_')
    coords['Variable'] = coords['Modalité'].str.split('_').str[0]

    # Tracer avec une couleur par variable
    fig = px.scatter(
        coords, x='Dim 1', y='Dim 2', text='Modalité', color='Variable',
        title=titre, width=800, height=700
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(
        xaxis=dict(title="Dimension 1"),
        yaxis=dict(title="Dimension 2"),
        showlegend=False
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


X = df_merged_selection.drop(columns=['% d’admis dont filles', 'classe_filles'])   # Variables explicatives
y = df_merged_selection['classe_filles']  # Cible
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#indices des variables catégorielles nominales
cat_indices = [X.columns.get_loc(col) for col in nominales]

clf = HistGradientBoostingClassifier(random_state=42,categorical_features=[cat_indices])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#print("Accuracy:", accuracy_score(y_test, y_pred))


# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)


# Calcule l'importance des features
result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# Met sous forme de Series pandas pour affichage
importances = pd.Series(result.importances_mean, index=X_test.columns)


importances_sorted = importances.sort_values(ascending=False)

def plot_importances_variables(importances_series, title="Importance des variables"):

    # Trier les importances
    importances_sorted = importances_series.sort_values(ascending=False)

    # Création de la figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=importances_sorted.index,
        y=importances_sorted.values,
        marker_color='skyblue',
        name='Importance',
    ))

    # Ajouter une ligne horizontale à y = 0
    fig.add_shape(
        type='line',
        x0=-0.5,
        x1=len(importances_sorted) - 0.5,
        y0=0,
        y1=0,
        line=dict(color='red', width=1),
    )

    # Mise en forme
    fig.update_layout(
        title=title,
        xaxis_title="Variables",
        yaxis_title="Importance moyenne",
        xaxis_tickangle=45,
        margin=dict(t=50, l=50, r=50, b=150),
        height=400,
        width=1200
    )

    return fig

#scraping github et google scholar


# Données scraping sur les profils github et google scholar
scraping_df = pd.read_csv("scraping_merged_dataset.csv")

# Nettoyage et renommage
scraping_df['gender'] = scraping_df['gender'].fillna("inconnu").replace({
    'female': 'femme',
    'male': 'homme',
    'unknown': 'inconnu',
    'Unknown': 'inconnu'
})
scraping_df['country'] = scraping_df['country'].fillna("inconnu")
scraping_df['tags'] = scraping_df['tags'].fillna("inconnu")

scraping_df = scraping_df.rename(columns={
    'gender': 'genre',
    'country': 'pays',
    'tags': 'domaines'
})

# Extraction des domaines
df_domaines = scraping_df[['genre', 'domaines']].copy()
df_domaines = df_domaines.assign(domaine=df_domaines['domaines'].str.lower().str.split(','))
df_domaines = df_domaines.explode('domaine')
df_domaines['domaine'] = df_domaines['domaine'].str.strip()



def plot_top_pays(df):
    top_countries = df['pays'].value_counts().head(10).reset_index()
    top_countries.columns = ['pays', 'count']
    fig = px.bar(top_countries, x='pays', y='count', color='pays',
                 title="Top 10 des pays représentés",
                 labels={"pays": "Pays", "count": "Nombre"},
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(showlegend=False, xaxis_tickangle=45)
    return fig

def plot_genre_par_domaine(df_domaines):
    top_domaines = df_domaines['domaine'].value_counts().head(9).index.tolist()
    filtered_tags_df = df_domaines[df_domaines['domaine'].isin(top_domaines)]
    fig = px.histogram(filtered_tags_df, x='domaine', color='genre',
                       category_orders={'domaine': top_domaines},
                       title="Répartition des genres dans les 9 domaines les plus fréquents",
                       labels={"domaine": "Domaine", "genre": "Genre"},
                       color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(xaxis_tickangle=45, yaxis_title="Nombre")
    return fig

def plot_ecart_genres_par_pays(df):
    grouped = df.groupby(['pays', 'genre']).size().reset_index(name='count')
    pivot = grouped.pivot(index='pays', columns='genre', values='count').fillna(0).astype(int)
    pivot['total'] = pivot.sum(axis=1)
    pivot['gender_gap'] = (pivot.get('homme', 0) - pivot.get('femme', 0)) / pivot['total']
    pivot['hover_text'] = pivot.apply(
        lambda row: f"<b>{row.name}</b><br>" + "<br>".join([
            f"{col}: {int(row[col])}" for col in pivot.columns if col not in ['total', 'hover_text', 'gender_gap']
        ]), axis=1)
    pivot = pivot.reset_index()

    fig = px.scatter_geo(pivot, locations="pays", locationmode="country names",
                         size="total", color="gender_gap", custom_data=["hover_text"],
                         projection="natural earth", title="Répartition des genres par pays",
                         size_max=50, color_continuous_scale="RdBu", range_color=[-1, 1])
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>",
                      marker=dict(line=dict(width=0.5, color='gray')))
    fig.update_layout(coloraxis_colorbar=dict(
        title="Écart F/H", tickvals=[-1, 0, 1], ticktext=["Femme", "Égal", "Homme"]
    ), showlegend=False)
    return fig

def plot_genre_par_source(df):
    if 'source' not in df.columns:
        return go.Figure().update_layout(title="Colonne 'source' non disponible")

    genre_par_source = df.groupby(['source', 'genre']).size().reset_index(name='count')
    genres_uniques = sorted(genre_par_source['genre'].unique())

    palette = pc.qualitative.Prism
    colors = {genre: palette[i % len(palette)] for i, genre in enumerate(genres_uniques)}

    sources = genre_par_source['source'].unique()
    n = len(sources)

    fig = make_subplots(rows=1, cols=n, specs=[[{'type': 'domain'}]*n])

    for i, source in enumerate(sources):
        data_source = genre_par_source[genre_par_source['source'] == source]
        data_source = data_source.set_index('genre').reindex(genres_uniques, fill_value=0).reset_index()
        
        fig.add_trace(
            go.Pie(
                labels=data_source['genre'],
                values=data_source['count'],
                name=source,
                hole=0.4,
                marker=dict(colors=[colors[g] for g in data_source['genre']]),
                textinfo='percent+label'
            ),
            row=1, col=i+1
        )

        domain = fig.data[-1].domain
        x_center = (domain.x[0] + domain.x[1]) / 2 # centre du subplot
        fig.add_annotation(
            x=x_center,
            y=-0.15,
            text=source,
            showarrow=False,
            xref='paper',
            yref='paper',
            font=dict(size=14, color="black"),
            align="center"
        )

    fig.update_layout(title_text="Répartition des genres par source", showlegend=True)
    return fig






st.write("# Écart de présence entre les genres dans les études supérieures en informatique puis dans le milieu professionnel et la recherche")


#Texte descriptif de l'outil
st.write("""
        Cet outil permet d'analyser les données des formations et des candidatures dans le domaine de l'informatique dans le milieu universitaire français de 2019 à 2024 (De 2023 à 2024 pour MonMaster). 
        Les données se concentrent sur des thèmes porteurs économiquement tels que la science des données, l'intelligence artificielle, la blockchain etc.
        L'objectif est d'étudier la répartition des candidats et des admis
        par genre. Les données proviennent de Parcoursup et de MonMaster, et sont préparées à l'aide d'un notebook dédié.
         
        À noter également que les formations étudiées sont celles qui sont sélectives (sauf pour l'année 2019 où l'information est manquante). Le traitement des données est détail dans des notebooks dédiés référencés en bas de page.
    """)









st.write("### Taux moyens d'admission des filles par académie et par année pour les formations informatiques (Parcoursup & MonMaster)")

st.plotly_chart(plot_taux_moyen_admission_filles_academie(parcoursup_data_merged,"Taux moyen d'admission des filles par académie et par année des formations Parcoursup"))
st.write("""
        Entre 2019 et 2024, les données de Parcoursup révèlent une progression notable de la part des filles admises dans les formations informatiques à travers plusieurs académies.
        Les académies de Paris, Versailles, Polynésie française, Toulouse et Reims se distinguent par les augmentations les plus significatives dans un domaine historiquement masculin.
        Les taux les plus élevés d'admises féminines sont observés dans les académies de Paris, Polynésie française, Mayotte, Versailles, Martinique, Créteil et Guyane.
        On remarque d'ailleurs que ces académies sont majoritairement situées en Île-de-France et dans les territoires d'outre-mer, ce qui pourrait traduire des dynamiques régionales spécifiques en faveur de la parité dans les formations informatiques.
""")
st.plotly_chart(plot_taux_moyen_admission_filles_academie(monmaster_data_merged,"Taux moyen d'admission des filles par académie et par année des formations MonMaster"))
st.write("""
        Entre 2023 et 2024, les formations en informatique proposées via MonMaster affichent une évolution encourageante en matière de parité, avec une hausse notable du taux d'admission des femmes dans plusieurs académies. 
        Les académies de Nancy-Metz, Nantes, Normandie, Reims, Strasbourg et Créteil enregistrent les plus fortes augmentations du taux d'admises.
        En 2024, les académies de Nancy-Metz, Strasbourg, Créteil, Normandie, Versailles, Toulouse et Paris présentent les taux d'admises les plus élevés. 
""")




st.write("### Effectif des admises par académie et par année pour les formations informatiques (Parcoursup & MonMaster)")

st.plotly_chart(plot_effectif_admises_academie(parcoursup_data_merged,"Effectif des admises par académie et par année des formations Parcoursup"))
st.write("""
        Entre 2019 et 2024, l'évolution de l'effectif des filles admises dans les formations informatiques via Parcoursup montre une progression encourageante. 
        Les académies de Versailles, La Réunion et Poitiers se démarquent par les plus fortes hausses du nombre d'admises, reflétant une dynamique positive en faveur de l'accès des jeunes femmes à ces filières.
        En termes d'effectifs absolus, les académies de Versailles, Paris, Créteil, Toulouse et Lyon concentrent les plus grands nombres d'admises.
         """)

st.plotly_chart(plot_effectif_admises_academie(monmaster_data_merged,"Effectif des admises par académie et par année des formations MonMaster"))
st.write("""
        Entre 2023 et 2024, l'évolution de l'effectif des filles admises dans les formations informatiques via MonMaster montre une progression encourageante. 
        Les académies de Créteil, Strasbourg et Paris se démarquent par les plus fortes hausses du nombre d'admises, reflétant une dynamique positive en matière de parité.
        En termes d'effectifs absolus, les académies de Paris, Toulouse, Strasbourg, Versailles et Créteil concentrent les plus grands nombres d'admises.
         """)




st.write("### Taux moyens d'admission des filles par filière pour les formations informatiques (Parcoursup & MonMaster)")

st.plotly_chart(plot_taux_moyen_admission_filles_filiere(parcoursup_data_merged,"Taux moyen d'admission des filles par filière des formations Parcoursup"))
st.write("""
        Proportionnellement le taux d'admises dans les formations informatiques via Parcoursup est plus important dans les filières générales (Licence et CPGE) que pour les filières professionnelles et technologiques.
""")

st.plotly_chart(plot_taux_moyen_admission_filles_filiere(monmaster_data_merged,"Taux moyen d'admission des filles par filière des formations MonMaster"))
st.write("""
        La répartition du taux d'admises dans les formations informatiques via MonMaster semble plutôt équilibré entre les universités et les écoles privées.
        L'absence de formations catégorisées comme école d'ingénieur s'explique par le fait que la majorité des candidatures pour des masters dans ces établissements ne se font pas sur MonMaster.
""")




st.write("### Évolution du taux d'admises par année pour les formations informatiques (Parcoursup & MonMaster)")

st.plotly_chart(plot_taux_admises_par_annee(parcoursup_data_merged,"Évolution du taux d'admises par année des formations Parcoursup"))
st.write("""
        Entre 2019 et 2024, l'évolution du taux d'admises dans les formations informatiques via Parcoursup est relativement stable (moins de 2% de différence entre les extrêmes).
""")

st.plotly_chart(plot_taux_admises_par_annee(monmaster_data_merged,"Évolution du taux d'admises par année des formations MonMaster"))
st.write("""
        Entre 2023 et 2024, l'évolution du taux d'admises par année dans les formations informatiques MonMaster est en très légère augmentation encourageante (+2%).
""")


st.write("### Analyse en composantes principales (ACP)")

st.plotly_chart(plot_cercle_correlation(pca.components_,colonnes_numeriques))
st.write("""L'ACP (Analyse en composantes principales) permet d'observer les corrélations entre variables quantitatives (avec des valeurs numériques). Plus l'angle entre deux flèches sont proches plus elles sont liées.
         Un angle de 0° indique une corrélation parfaite, un angle de 90° indique une absence de corrélation et enfin un angle de 180° indique une corrélation négative parfaite.
         Ici, on a une faible corrélation positive entre le pourcentage d'admises et l'effectif de candidates ce qui semble cohérent. On remarque également que l'effectif de total des candidats et l'effectif des candidats garçons sont décorrélés du pourcentage d'admises. Enfin, il semble y avoir une corrélation négative entre le pourcentage d'admises et le pourcentage d'admis provenant d'un établissement issu de la même académie. Ce qui pourrait suggérer que les filles ne choisissent pas en priorité des établissements de leur académie. 
         """)


st.write("### Répartition des classes de filles admises par année pour les formations informatiques (Parcoursup & MonMaster)")
st.write("""
    Les trois classes "faible, moyen et élevé" sont définies à partir des quantiles selon le pourcentage de filles admises.

""")

st.plotly_chart(plot_repartition_classes_filles(parcoursup_data_classe,"Répartition des classes de filles admises par année pour les formations informatiques Parcoursup"))
st.write("""
        Entre 2019 et 2024, la part de classe d'admise "faible" des formations informatiques Parcoursup semble en légère augmentation, la part de classe d'admise "moyenne" reste plutôt constante.
        On note une belle évolution de la part de classe d'admise "élevé" (+40%).
""")

st.plotly_chart(plot_repartition_classes_filles(monmaster_data_classe,"Répartition des classes de filles admises par année pour les formations informatiques MonMaster"))
st.write("""
    Entre 2023 et 2024, l'évolution de la répartition des classes de filles admises reste plutôt stable dans l'ensemble et équilibrée.
""")


st.write("### Analyse des correspondances multiples (ACM)")

st.plotly_chart(plot_acm(df_merged_acm,colonnes_acm))
st.write("""
        L'ACM (Analyse des Correspondances Multiples) permet de visualiser des proximités entre modalités d'une variable en réduisant les dimensions.
        Si on s'intéresse à la classe d'admise "élevé" soit la classe 2, on remarque que les académies les plus proches sont Strasbourg, Toulouse et Paris. 
        Aussi par rapport à la dimension 1 (en abscisse), la classe 2 est proche de la filière licence et des années 2023 et 2024. 
        Par rapport à la dimension 2 (en ordonnées), le niveau de la formation le plus proche est Bac +5 et la filière Université ce qui semble correspondre aux master publics. 
        Il semble se dessiner un profil pour lequel le taux d'admises est le plus grand: les formations universitaires (Licence, Master) en 2023 et 2024 avec en top académies Paris, Strasbourg et Toulouse.
        """)


st.write("## Analyse complémentaire : Répartition des profils scraping (présence en ligne)")

st.write("### 1. Pays les plus représentés en ligne")
st.plotly_chart(plot_top_pays(scraping_df))

st.write("### 2. Répartition des genres dans les principaux domaines")
st.plotly_chart(plot_genre_par_domaine(df_domaines))

st.write("### 3. Répartition des genres par pays")
st.plotly_chart(plot_ecart_genres_par_pays(scraping_df))

st.write("### 4. Répartition des genres par source")
st.plotly_chart(plot_genre_par_source(scraping_df))





# Afficher chaque dataset avec un titre et les informations supplémentaires
st.write("### Données Parcoursup en informatique")
st.dataframe(parcoursup_data_merged)

st.write("### Données MonMaster en informatique")
st.dataframe(monmaster_data_merged)

st.write("### Données Parcoursup et MonMaster en informatique")
st.dataframe(df_merged)



st.write("Source des données Parcoursup: [Parcoursup](https://data.enseignementsup-recherche.gouv.fr/pages/parcoursupdata/?disjunctive.fili)")
st.write("Source des données MonMaster : [MonMaster](https://www.data.gouv.fr/fr/datasets/monmaster-2024-voeux-de-poursuite-detudes-et-de-reorientation-en-master-et-reponses-des-etablissements/)")

def display_source_parcoursup(year):
    st.write(f"Méthode de préparation des données Parcoursup {year}: [Notebook](https://www.kaggle.com/code/alexismouroux/parcoursup-analysis-{year})")

def display_source_monmaster(year):
    st.write(f"Méthode de préparation des données MonMaster {year}: [Notebook](https://www.kaggle.com/code/alexismouroux/monmaster-analysis-{year})")

for year, df in zip(range(2019, 2025), [parcoursup2019, parcoursup2020, parcoursup2021, parcoursup2022, parcoursup2023, parcoursup2024]):
    display_source_parcoursup(year)

for year, df in zip(range(2023, 2025), [monmaster2023, monmaster2024]):
    display_source_monmaster(year)



st.write(f"Méthode de préparation des données Parcoursup et MonMaster {year}: [Notebook](https://www.kaggle.com/code/alexismouroux/merge-parcoursup-monmaster-datasets)")