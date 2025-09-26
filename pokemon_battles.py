import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os

from pathlib import Path

# =========================
# CSS för UI
# =========================
st.markdown(
    """
    <style>
    .poke-card { display:flex; flex-direction:column; align-items:center; }
    .poke-img-box { width:100%; height:260px; display:flex; align-items:center; justify-content:center; background:rgba(255,255,255,0.03); border-radius:12px; }
    .poke-img-box img { max-height:240px; max-width:100%; object-fit:contain; }
    .badges { display:flex; gap:8px; margin-top:6px; margin-bottom:14px; }
    .badge { padding:6px 10px; border-radius:8px; color:#ffffff; font-weight:600; }
    </style>a
    """,
    unsafe_allow_html=True
)

# =========================
# Konstanter
# =========================

BASE = Path(__file__).resolve().parent

POKEMON_CSV = BASE / "dataset" / "pokemon.csv"
RF_MODEL_PKL = BASE / "pokemon_battle_rf.pkl"       # se modellhantering nedan
FEATURE_COLS_PKL = BASE / "feature_columns.pkl"

NAME_IMG_URL = "https://img.pokemondb.net/artwork/large/{slug}.jpg"

# =========================
# Hjälpfunktioner
# =========================
@st.cache_data
def load_pokemon():
    df = pd.read_csv(POKEMON_CSV)

    df["Name"] = df["Name"].astype(str)
    df.loc[df["Name"].str.strip().isin(["", "nan", "NaN"]), "Name"] = np.nan

    missing_names = {63: "Primeape"}
    df["Name"] = df.apply(lambda r: missing_names.get(r["#"], r["Name"]), axis=1)

    # Droppa ev. kvarvarande utan namn
    df = df.dropna(subset=["Name"]).copy()

    # Legendary -> 0/1
    if df["Legendary"].dtype == bool:
        df["Legendary"] = df["Legendary"].astype(int)
    else:
        df["Legendary"] = (
            df["Legendary"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)
        )

    # Gör om # till pokemon_id
    df = df.rename(columns={"#": "pokemon_id"})

    for col in ["Type 1", "Type 2"]:
        df[col] = (
            df[col].astype(str).str.strip()
            .replace({"": "None", "nan": "None", "NaN": "None"})
            .fillna("None")
        )

    stats_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Generation", "Legendary"]
    types_df = df[["pokemon_id", "Type 1", "Type 2"]].copy()
    return df, types_df, stats_cols

@st.cache_resource
def load_model():
    
    if RF_MODEL_PKL.exists() and FEATURE_COLS_PKL.exists():
        model = joblib.load(RF_MODEL_PKL)
        feat_cols = joblib.load(FEATURE_COLS_PKL)
        return model, feat_cols
    
    # Laddar in pkl-filerna från Google Drive
    MODEL_FILE_ID = "1uOBsEiWgJdRAc2KKK5DnBoFTLLT26HfB"
    FEATURES_FILE_ID = "1tiX20za2GnXhG-jjcZ8LP1gjCkA2-92T"
    
    # Ladda ned från Google Drive
    for file_id, filename in [(MODEL_FILE_ID, "temp_model.pkl"), (FEATURES_FILE_ID, "temp_features.pkl")]:
        if not os.path.exists(filename):
            st.info(f"Laddar {filename}...")
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(response.content)
    
    # Ladda modellerna
    model = joblib.load("temp_model.pkl")
    feat_cols = joblib.load("temp_features.pkl")
    
    return model, feat_cols

def slugify_pokemon_name(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("♀", "-f").replace("♂", "-m")
    s = s.replace(".", "").replace("’", "").replace("'", "")
    s = s.replace("é", "e")
    s = s.replace(" ", "-")
    special = {
        "mr mime": "mr-mime",
        "mime jr": "mime-jr",
        "farfetch'd": "farfetchd",
        "type: null": "type-null",
        "nidoran♀": "nidoran-f",
        "nidoran♂": "nidoran-m",
        "ho-oh": "ho-oh",
    }
    return special.get(s, s)

@st.cache_data(show_spinner=False)
def image_url_by_name(poke_name: str) -> str:
    slug = slugify_pokemon_name(poke_name)
    return NAME_IMG_URL.format(slug=slug)

# =========================
# Ladda resurser
# =========================
pokemon_min, types_df, STATS = load_pokemon()
model, FEATURE_COLS = load_model()

# =========================
# Sidebar: filter
# =========================
st.sidebar.header("Inställningar")
st.sidebar.markdown("**Modell:** Random Forest")

# Filtrering: Generation
gen_values = sorted(pokemon_min["Generation"].dropna().astype(int).unique().tolist())
gen_option = st.sidebar.selectbox("Generation", ["Alla"] + gen_values, index=0)

# Filtrering: Typer
all_types = pd.unique(
    pd.concat([pokemon_min["Type 1"].astype(str), pokemon_min["Type 2"].astype(str)])
).tolist()
all_types = sorted([t for t in all_types if t and t != "nan" and t != "None"])
type_selection = st.sidebar.multiselect("Filtrera på typ(er)", all_types, default=[])

# Sök
name_query = st.sidebar.text_input("Sök namn", value="")

names_df = pokemon_min[["pokemon_id", "Name", "Type 1", "Type 2", "Generation"]].copy()
if gen_option != "Alla":
    names_df = names_df[names_df["Generation"].astype(int) == int(gen_option)]
if type_selection:
    names_df = names_df[(names_df["Type 1"].isin(type_selection)) | (names_df["Type 2"].isin(type_selection))]
if name_query:
    names_df = names_df[names_df["Name"].str.contains(name_query, case=False, na=False)]

names_df = names_df.dropna(subset=["Name"])  # säkra upp
if names_df.empty:
    names_df = pokemon_min[["pokemon_id", "Name", "Type 1", "Type 2", "Generation"]].copy()
    names_df = names_df.dropna(subset=["Name"])

id_to_name = dict(zip(names_df["pokemon_id"], names_df["Name"]))
name_to_id = {v: k for k, v in id_to_name.items()}
all_names = sorted(name_to_id.keys(), key=lambda x: str(x))

# =========================
# Små helpers för UI
# =========================
TYPE_COLORS = {
    "Normal": "#A8A77A","Fire": "#EE8130","Water": "#6390F0","Electric": "#F7D02C","Grass": "#7AC74C",
    "Ice": "#96D9D6","Fighting": "#C22E28","Poison": "#A33EA1","Ground": "#E2BF65","Flying": "#A98FF3",
    "Psychic": "#F95587","Bug": "#A6B91A","Rock": "#B6A136","Ghost": "#735797","Dragon": "#6F35FC",
    "Dark": "#705746","Steel": "#B7B7CE","Fairy": "#D685AD","None": "#9AA0A6"
}

def render_types(pid: int):
    row = types_df.loc[types_df["pokemon_id"] == pid]
    t1 = str(row["Type 1"].iat[0]) if not row.empty else "None"
    t2 = str(row["Type 2"].iat[0]) if not row.empty else "None"
    t1 = "None" if t1.lower() == "nan" or t1 == "" else t1
    t2 = "None" if t2.lower() == "nan" or t2 == "" else t2
    html = "<div class='badges'>"
    if t1 != "None":
        html += f"<div class='badge' style='background:{TYPE_COLORS.get(t1,'#999')}'>{t1}</div>"
    if t2 != "None":
        html += f"<div class='badge' style='background:{TYPE_COLORS.get(t2,'#999')}'>{t2}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_card(name: str, pid: int, img_url: str):
    st.markdown(
        f"""
        <div class='poke-card'>
          <div class='poke-img-box'>
            <img src='{img_url}' alt='{name}' />
          </div>
          <div style='text-align:center; margin-top:6px;'>{name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_types(pid)

# =========================
# UI
# =========================
st.title("Pokémon Battle Predictor")
st.caption("Tränad på combats + stats + typer. Välj två Pokémon och få en prediktion.")

colm = st.columns(2)
with colm[0]:
    default_1 = all_names.index("Charizard") if "Charizard" in all_names else 0
    p1_name = st.selectbox("Pokémon 1", all_names, index=default_1)
    p1_id = name_to_id[p1_name]
    render_card(p1_name, p1_id, image_url_by_name(p1_name))

with colm[1]:
    default_2 = all_names.index("Venusaur") if "Venusaur" in all_names else 1
    p2_name = st.selectbox("Pokémon 2", all_names, index=default_2)
    p2_id = name_to_id[p2_name]
    render_card(p2_name, p2_id, image_url_by_name(p2_name))

# =========================
# Feature builder
# =========================
def build_row_for_ids(p1_id: int, p2_id: int) -> pd.DataFrame:
    row = {}

    p1 = pokemon_min.loc[pokemon_min["pokemon_id"] == p1_id, STATS]
    p2 = pokemon_min.loc[pokemon_min["pokemon_id"] == p2_id, STATS]

    for c in STATS:
        row[f"diff_{c}"] = float(p1[c].values[0] - p2[c].values[0])

    t1 = types_df.loc[types_df["pokemon_id"] == p1_id, ["Type 1", "Type 2"]]
    t2 = types_df.loc[types_df["pokemon_id"] == p2_id, ["Type 1", "Type 2"]]
    t1_1 = str((t1["Type 1"].iloc[0] if not t1.empty else "None") or "None")
    t1_2 = str((t1["Type 2"].iloc[0] if not t1.empty else "None") or "None")
    t2_1 = str((t2["Type 1"].iloc[0] if not t2.empty else "None") or "None")
    t2_2 = str((t2["Type 2"].iloc[0] if not t2.empty else "None") or "None")

    row_df = pd.DataFrame([row])
    type_frame = pd.DataFrame([{
        "P1_Type 1": t1_1, "P1_Type 2": t1_2,
        "P2_Type 1": t2_1, "P2_Type 2": t2_2
    }])

    dummies = pd.get_dummies(
        type_frame,
        prefix=["P1_T1", "P1_T2", "P2_T1", "P2_T2"],
        columns=["P1_Type 1", "P1_Type 2", "P2_Type 1", "P2_Type 2"]
    )

    out = pd.concat([row_df, dummies], axis=1)

    for c in FEATURE_COLS:
        if c not in out.columns:
            out[c] = 0
    out = out[FEATURE_COLS]
    return out

# =========================
# Prediktion
# =========================
if st.button("Predicera vinnare"):
    if p1_id == p2_id:
        st.warning("Välj två olika Pokémon.")
    else:
        X_row = build_row_for_ids(p1_id, p2_id)
        pred = int(model.predict(X_row)[0])

        p1 = id_to_name[p1_id]
        p2 = id_to_name[p2_id]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_row)[0]
            proba_txt = f"Sannolikhet P1 vinner: {proba[1]:.3f}"
        else:
            proba_txt = ""

        if pred == 1:
            st.success(f"Prediktion: {p1} vinner")
            if proba_txt: st.write(proba_txt)
        else:
            st.success(f"Prediktion: {p2} vinner")
            if proba_txt: st.write(proba_txt)