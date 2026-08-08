# -*- coding: utf-8 -*-
"""
=============================================================================
 HIGH-THROUGHPUT MATERIAL SCREENING ENGINE - A2BB'O6 | V5.0 PRODUCTION-GRADE
 Inclus : Dynamique UI, QE .scf.in, Ordre B/B', Descripteurs ML/DeepXDE
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import zipfile
import plotly.express as px
import time
from typing import Dict, Optional

# ==============================================================================
# 1. BASE DE DONNÉES LOCALE ULTIME (Shannon + Pauling + Propriétés)
# ==============================================================================
ELEMENT_DB = {
    "O":  {"radii": {-2: {2: 1.35, 3: 1.36, 4: 1.38, 6: 1.40, 8: 1.42, 12: 1.44}}, "chi_pauling": 3.44, "group": 16, "mass": 15.999},
    # Site A
    "Ca": {"radii": {2: {6: 1.00, 8: 1.12, 12: 1.34}}, "chi_pauling": 1.00, "group": 2, "mass": 40.078},
    "Sr": {"radii": {2: {6: 1.18, 8: 1.26, 12: 1.44}}, "chi_pauling": 0.95, "group": 2, "mass": 87.62},
    "Ba": {"radii": {2: {6: 1.35, 8: 1.42, 12: 1.61}}, "chi_pauling": 0.89, "group": 2, "mass": 137.327},
    "Pb": {"radii": {2: {6: 1.19, 8: 1.29, 12: 1.49}}, "chi_pauling": 2.33, "group": 14, "mass": 207.2},
    "La": {"radii": {3: {6: 1.032, 8: 1.16, 12: 1.36}}, "chi_pauling": 1.10, "group": 3, "mass": 138.905},
    "Ce": {"radii": {3: {6: 1.01, 8: 1.143, 12: 1.34}}, "chi_pauling": 1.12, "group": 3, "mass": 140.116},
    "Pr": {"radii": {3: {6: 0.99, 8: 1.126, 12: 1.32}}, "chi_pauling": 1.13, "group": 3, "mass": 140.908},
    "Nd": {"radii": {3: {6: 0.983, 8: 1.109, 12: 1.31}}, "chi_pauling": 1.14, "group": 3, "mass": 144.242},
    "Sm": {"radii": {3: {6: 0.958, 8: 1.079, 12: 1.28}}, "chi_pauling": 1.17, "group": 3, "mass": 150.36},
    "Bi": {"radii": {3: {6: 1.03, 8: 1.17, 12: 1.38}}, "chi_pauling": 2.02, "group": 15, "mass": 208.980},
    # Sites B et B'
    "Sc": {"radii": {3: {6: 0.745}}, "chi_pauling": 1.36, "group": 3, "mass": 44.956},
    "Ti": {"radii": {3: {6: 0.67}, 4: {6: 0.605}}, "chi_pauling": 1.54, "group": 4, "mass": 47.867},
    "V":  {"radii": {3: {6: 0.64}, 4: {6: 0.58}, 5: {6: 0.54}}, "chi_pauling": 1.63, "group": 5, "mass": 50.941},
    "Cr": {"radii": {3: {6: 0.615}, 4: {6: 0.55}}, "chi_pauling": 1.66, "group": 6, "mass": 51.996},
    "Mn": {"radii": {2: {6: 0.83}, 3: {6: 0.645}, 4: {6: 0.53}}, "chi_pauling": 1.55, "group": 7, "mass": 54.938},
    "Fe": {"radii": {2: {6: 0.78}, 3: {6: 0.645}, 4: {6: 0.585}}, "chi_pauling": 1.83, "group": 8, "mass": 55.845},
    "Co": {"radii": {2: {6: 0.745}, 3: {6: 0.61}, 4: {6: 0.53}}, "chi_pauling": 1.88, "group": 9, "mass": 58.933},
    "Ni": {"radii": {2: {6: 0.69}, 3: {6: 0.60}}, "chi_pauling": 1.91, "group": 10, "mass": 58.693},
    "Cu": {"radii": {2: {6: 0.73}}, "chi_pauling": 1.90, "group": 11, "mass": 63.546},
    "Zn": {"radii": {2: {6: 0.74}}, "chi_pauling": 1.65, "group": 12, "mass": 65.38},
    "Zr": {"radii": {4: {6: 0.72}}, "chi_pauling": 1.33, "group": 4, "mass": 91.224},
    "Nb": {"radii": {3: {6: 0.72}, 4: {6: 0.68}, 5: {6: 0.64}}, "chi_pauling": 1.60, "group": 5, "mass": 92.906},
    "Mo": {"radii": {3: {6: 0.69}, 4: {6: 0.65}, 5: {6: 0.61}, 6: {6: 0.59}}, "chi_pauling": 2.16, "group": 6, "mass": 95.95},
    "Ru": {"radii": {4: {6: 0.62}, 5: {6: 0.565}}, "chi_pauling": 2.20, "group": 8, "mass": 101.07},
    "Rh": {"radii": {3: {6: 0.665}, 4: {6: 0.60}}, "chi_pauling": 2.28, "group": 9, "mass": 102.905},
    "Hf": {"radii": {4: {6: 0.71}}, "chi_pauling": 1.30, "group": 4, "mass": 178.49},
    "Ta": {"radii": {4: {6: 0.68}, 5: {6: 0.64}}, "chi_pauling": 1.50, "group": 5, "mass": 180.948},
    "W":  {"radii": {4: {6: 0.66}, 5: {6: 0.62}, 6: {6: 0.60}}, "chi_pauling": 2.36, "group": 6, "mass": 183.84},
    "Re": {"radii": {4: {6: 0.63}, 5: {6: 0.58}, 6: {6: 0.55}}, "chi_pauling": 1.90, "group": 7, "mass": 186.207},
    "Os": {"radii": {4: {6: 0.63}, 5: {6: 0.575}, 6: {6: 0.55}}, "chi_pauling": 2.20, "group": 8, "mass": 190.23},
    "Ir": {"radii": {3: {6: 0.68}, 4: {6: 0.625}, 5: {6: 0.57}}, "chi_pauling": 2.20, "group": 9, "mass": 192.217},
    "Pt": {"radii": {4: {6: 0.625}}, "chi_pauling": 2.28, "group": 10, "mass": 195.084},
    "Al": {"radii": {3: {6: 0.535}}, "chi_pauling": 1.61, "group": 13, "mass": 26.981},
    "Ga": {"radii": {3: {6: 0.62}}, "chi_pauling": 1.81, "group": 13, "mass": 69.723},
    "Mg": {"radii": {2: {6: 0.72}}, "chi_pauling": 1.31, "group": 2, "mass": 24.305},
}

# ==============================================================================
# 2. CONFIGURATION DYNAMIQUE DES STRUCTURES AVEC PARAMÈTRES PAR DÉFAUT
# ==============================================================================
STRUCTURE_CONFIG = {
    "Double Pérovskite": {
        "Cubique (Fm-3m)": {"sg": "Fm-3m", "cn_A": 12, "lattice": "cubic", "default_a": 8.0, "default_delta_a": 0.5},
        "Tétragonale (I4/m)": {"sg": "I4/m", "cn_A": 8, "lattice": "tetra", "default_a": 5.6, "default_delta_a": 0.5},
        "Orthorhombique (Pnma)": {"sg": "Pnma", "cn_A": 8, "lattice": "ortho", "default_a": 5.5, "default_delta_a": 0.3},
        "Monoclinique (P2_1/n)": {"sg": "P2_1/n", "cn_A": 8, "lattice": "mono", "default_a": 5.5, "default_delta_a": 0.3}
    },
    "Corindon / Ilménite": {
        "Rhomboédrique (R-3)": {"sg": "R-3", "cn_A": 6, "lattice": "hexagonal", "default_a": 5.0, "default_delta_a": 0.3}
    }
}

# ==============================================================================
# 3. MOTEUR VECTORISÉ ET CRISTALLOGRAPHIQUE AVANCÉ
# ==============================================================================
class HTEngine:
    @staticmethod
    def extract_cations(cn_A: int, forbidden: list) -> tuple:
        cations_A, cations_B = [], []
        for el, props in ELEMENT_DB.items():
            if el in forbidden or el == "O": continue
            for ox, coord_dict in props["radii"].items():
                if ox > 0:
                    if cn_A in coord_dict:
                        cations_A.append([el, ox, coord_dict[cn_A], props["chi_pauling"], props["group"], props["mass"]])
                    if 6 in coord_dict:
                        cations_B.append([el, ox, coord_dict[6], props["chi_pauling"], props["group"], props["mass"]])
        return cations_A, cations_B

    @staticmethod
    def vectorized_screening(cations_A: list, cations_B: list, r_O: float, target_a: float, delta_a: float, t_min: float, t_max: float, max_delta_chi: float, lattice_type: str) -> pd.DataFrame:
        start_time = time.time()
        
        df_A = pd.DataFrame(cations_A, columns=['el_A', 'ox_A', 'r_A', 'chi_A', 'grp_A', 'mass_A'])
        df_B = pd.DataFrame(cations_B, columns=['el_B', 'ox_B', 'r_B', 'chi_B', 'grp_B', 'mass_B'])
        
        df_B_left = df_B.copy()
        df_B_right = df_B.rename(columns={
            'el_B': 'el_Bp', 'ox_B': 'ox_Bp', 'r_B': 'r_Bp', 
            'chi_B': 'chi_Bp', 'grp_B': 'grp_Bp', 'mass_B': 'mass_Bp'
        })
        
        df_B_left['key'] = 1
        df_B_right['key'] = 1
        
        df_BxB = pd.merge(df_B_left, df_B_right, on='key')
        df_BxB = df_BxB[df_BxB['el_B'] < df_BxB['el_Bp']]
        
        df_BxB['req_2_ox_A'] = 12 - (df_BxB['ox_B'] + df_BxB['ox_Bp'])
        df_BxB = df_BxB[(df_BxB['req_2_ox_A'] > 0) & (df_BxB['req_2_ox_A'] % 2 == 0)]
        df_BxB['req_ox_A'] = df_BxB['req_2_ox_A'] / 2
        
        df_A_key = df_A[['el_A', 'ox_A', 'r_A', 'chi_A', 'grp_A', 'mass_A']].copy()
        df_comb = pd.merge(df_BxB, df_A_key, left_on='req_ox_A', right_on='ox_A')
        
        r_B_eff = (df_comb['r_B'] + df_comb['r_Bp']) / 2.0
        
        df_comb['t'] = (df_comb['r_A'] + r_O) / (np.sqrt(2) * (r_B_eff + r_O))
        df_comb = df_comb[(df_comb['t'] >= t_min) & (df_comb['t'] <= t_max)]
        
        df_comb['delta_chi'] = np.abs(df_comb['chi_B'] - df_comb['chi_Bp'])
        df_comb = df_comb[df_comb['delta_chi'] <= max_delta_chi]
        
        delta_r_BBp = np.abs(df_comb['r_B'] - df_comb['r_Bp'])
        delta_z_BBp = np.abs(df_comb['ox_B'] - df_comb['ox_Bp'])
        df_comb['order_propensity'] = (0.5 * (delta_r_BBp / 0.15) + 0.5 * (delta_z_BBp / 2.0)).clip(0.0, 1.0)
        
        a_A = np.sqrt(2) * (df_comb['r_A'] + r_O)
        a_B = 2 * (r_B_eff + r_O) * np.sqrt(2)
        df_comb['a_c'] = (a_A + a_B) / 2.0
        
        if lattice_type == "cubic":
            df_comb['a_calc'] = df_comb['a_c']
            df_comb['b_calc'] = df_comb['a_c']
            df_comb['c_calc'] = df_comb['a_c']
            df_comb['beta_calc'] = 90.0
        elif lattice_type == "tetra":
            df_comb['a_calc'] = df_comb['a_c']
            df_comb['b_calc'] = df_comb['a_c']
            df_comb['c_calc'] = df_comb['a_c'] * 1.02
            df_comb['beta_calc'] = 90.0
        elif lattice_type == "ortho":
            df_comb['a_calc'] = df_comb['a_c'] * np.sqrt(2)
            df_comb['b_calc'] = df_comb['a_c'] * 2
            df_comb['c_calc'] = df_comb['a_c'] * np.sqrt(2)
            df_comb['beta_calc'] = 90.0
        elif lattice_type == "mono":
            df_comb['a_calc'] = df_comb['a_c'] * np.sqrt(2)
            df_comb['b_calc'] = df_comb['a_c']
            df_comb['c_calc'] = df_comb['a_c'] * np.sqrt(2)
            df_comb['beta_calc'] = 135.0
        elif lattice_type == "hexagonal":
            df_comb['a_calc'] = 2 * (r_B_eff + r_O)
            df_comb['b_calc'] = df_comb['a_calc']
            df_comb['c_calc'] = df_comb['a_calc'] * np.sqrt(6)
            df_comb['beta_calc'] = 90.0

        df_comb = df_comb[np.abs(df_comb['a_calc'] - target_a) <= delta_a]
        
        mu = r_B_eff / r_O
        score_t = np.exp(-5.0 * np.abs(1.0 - df_comb['t']))
        score_mu = np.exp(-5.0 * np.abs(0.85 - mu))
        df_comb['stability_score'] = (0.7 * (score_t * score_mu) + 0.3 * df_comb['order_propensity']).round(3)
        
        df_comb['Formule'] = df_comb['el_A'] + '2' + df_comb['el_B'] + df_comb['el_Bp'] + 'O6'
        df_comb['d_e(B)'] = np.clip(df_comb['grp_B'] - df_comb['ox_B'], 0, 10)
        df_comb['d_e(Bp)'] = np.clip(df_comb['grp_Bp'] - df_comb['ox_Bp'], 0, 10)
        
        mass_O = ELEMENT_DB['O']['mass']
        chi_O = ELEMENT_DB['O']['chi_pauling']
        df_comb['mean_atomic_mass'] = (2 * df_comb['mass_A'] + df_comb['mass_B'] + df_comb['mass_Bp'] + 6 * mass_O) / 10.0
        df_comb['mean_chi'] = (2 * df_comb['chi_A'] + df_comb['chi_B'] + df_comb['chi_Bp'] + 6 * chi_O) / 10.0
        
        cols = ['Formule', 'el_A', 'el_B', 'el_Bp', 'ox_A', 'ox_B', 'ox_Bp',
                't', 'a_calc', 'b_calc', 'c_calc', 'beta_calc', 'stability_score', 
                'order_propensity', 'delta_chi', 'd_e(B)', 'd_e(Bp)', 'mean_atomic_mass', 'mean_chi']
        
        df_final = df_comb[cols].copy()
        df_final.rename(columns={'ox_A': 'Ox_A', 'ox_B': 'Ox_B', 'ox_Bp': 'Ox_Bp'}, inplace=True)
        
        st.session_state.exec_time = time.time() - start_time
        return df_final.reset_index(drop=True)

# ==============================================================================
# 4. GÉNÉRATEURS DE FICHIERS EXPERTS
# ==============================================================================
def generate_cif(row: pd.Series, sg: str) -> str:
    formula = row['Formule']
    a, b, c, beta = row['a_calc'], row['b_calc'], row['c_calc'], row['beta_calc']
    wyckoff = f"""
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
A1 {row['el_A']} 0.25 0.25 0.25
A2 {row['el_A']} 0.75 0.75 0.75
B1 {row['el_B']} 0.0 0.0 0.0
Bp1 {row['el_Bp']} 0.5 0.5 0.5
O1 O 0.25 0.0 0.0
O2 O 0.0 0.25 0.0
O3 O 0.0 0.0 0.25
O4 O 0.75 0.5 0.5
O5 O 0.5 0.75 0.5
O6 O 0.5 0.5 0.75
"""
    return f"""data_{formula}
_chemical_formula_sum '{formula}'
_symmetry_space_group_name_H-M '{sg}'
_cell_length_a {a:.4f}
_cell_length_b {b:.4f}
_cell_length_c {c:.4f}
_cell_angle_alpha 90.0
_cell_angle_beta {beta:.4f}
_cell_angle_gamma 90.0
{wyckoff}
"""

def generate_poscar(row: pd.Series) -> str:
    a, b, c, beta = row['a_calc'], row['b_calc'], row['c_calc'], row['beta_calc']
    beta_rad = np.deg2rad(beta)
    ax, bx, by = a, 0.0, b
    cx, cy, cz = c * np.cos(beta_rad), 0.0, c * np.sin(beta_rad)
    
    return f"""{row['Formule']}
1.0
{ax:.6f} 0.000000 0.000000
{bx:.6f} {by:.6f} 0.000000
{cx:.6f} {cy:.6f} {cz:.6f}
{row['el_A']} {row['el_B']} {row['el_Bp']} O
2 1 1 6
Direct
0.250000 0.250000 0.250000
0.750000 0.750000 0.750000
0.000000 0.000000 0.000000
0.500000 0.500000 0.500000
0.250000 0.000000 0.000000
0.000000 0.250000 0.000000
0.000000 0.000000 0.250000
0.750000 0.500000 0.500000
0.500000 0.750000 0.500000
0.500000 0.500000 0.750000
"""

def generate_qe_input(row: pd.Series) -> str:
    formula = row['Formule']
    a, b, c, beta = row['a_calc'], row['b_calc'], row['c_calc'], row['beta_calc']
    beta_rad = np.deg2rad(beta)
    ax, bx, by = a, 0.0, b
    cx, cy, cz = c * np.cos(beta_rad), 0.0, c * np.sin(beta_rad)
    el_A, el_B, el_Bp = row['el_A'], row['el_B'], row['el_Bp']
    
    return f"""&control
  calculation = 'scf'
  restart_mode = 'from_scratch'
  prefix = '{formula}'
  pseudo_dir = './pseudo/'
  outdir = './out/'
/
&system
  ibrav = 0, nat = 10, ntyp = 4,
  ecutwfc = 60.0,
  ecutrho = 480.0,
  occupations = 'smearing',
  smearing = 'm-p',
  degauss = 0.02
/
&electrons
  conv_thr = 1.0d-8
  mixing_beta = 0.4
/
CELL_PARAMETERS angstrom
  {ax:.6f}  0.000000  0.000000
  {bx:.6f}  {by:.6f}  0.000000
  {cx:.6f}  {cy:.6f}  {cz:.6f}

ATOMIC_SPECIES
  {el_A}  {ELEMENT_DB[el_A]['mass']}  {el_A}.pbe-spn-kjpaw_psl.1.0.0.UPF
  {el_B}  {ELEMENT_DB[el_B]['mass']}  {el_B}.pbe-spn-kjpaw_psl.1.0.0.UPF
  {el_Bp} {ELEMENT_DB[el_Bp]['mass']} {el_Bp}.pbe-spn-kjpaw_psl.1.0.0.UPF
  O   15.9990  O.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS crystal
  {el_A}   0.250000  0.250000  0.250000
  {el_A}   0.750000  0.750000  0.750000
  {el_B}   0.000000  0.000000  0.000000
  {el_Bp}  0.500000  0.500000  0.500000
  O    0.250000  0.000000  0.000000
  O    0.000000  0.250000  0.000000
  O    0.000000  0.000000  0.250000
  O    0.750000  0.500000  0.500000
  O    0.500000  0.750000  0.500000
  O    0.500000  0.500000  0.750000

K_POINTS automatic
  4 4 4 1 1 0
"""

# ==============================================================================
# 5. INTERFACE UTILISATEUR (STREAMLIT UI/UX)
# ==============================================================================
def main():
    st.set_page_config(page_title="⚛️ HTS A2BB'O6 Production V5", layout="wide", initial_sidebar_state="expanded")
    st.markdown("<style> .stApp { background-color: #0e1117; color: #fafafa; } .stButton>button { border-radius: 8px; transition: all 0.3s ease; } .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); } </style>", unsafe_allow_html=True)
    
    st.title("⚛️ Production-Grade HTS A₂BB'O6 Engine & ML Toolkit")
    st.caption("Vectorisation Pandas | Analyse d'Ordre B/B' | Quantum ESPRESSO (.in) | Export DeepXDE / ML")

    with st.sidebar:
        st.header("⚙️ Configuration Avancée")
        struct_type = st.selectbox("🏗️ Type de Structure", list(STRUCTURE_CONFIG.keys()))
        available_families = list(STRUCTURE_CONFIG[struct_type].keys())
        struct_family = st.selectbox("🔬 Famille / Groupe d'espace", available_families)
        
        config = STRUCTURE_CONFIG[struct_type][struct_family]
        st.info(f"Coordination Site A dynamique : **CN = {config['cn_A']}**\nGéométrie : **{config['lattice']}**")
        
        st.markdown("---")
        st.subheader("📏 Cible Paramètre de Maille (a)")
        
        # LOGIQUE DYNAMIQUE : Mise à jour intelligente des paramètres par défaut
        if 'last_struct_family' not in st.session_state:
            st.session_state.last_struct_family = struct_family
            
        if st.session_state.last_struct_family != struct_family:
            st.session_state.target_a = config['default_a']
            st.session_state.delta_a = config['default_delta_a']
            st.session_state.last_struct_family = struct_family
            
        if 'target_a' not in st.session_state:
            st.session_state.target_a = config['default_a']
        if 'delta_a' not in st.session_state:
            st.session_state.delta_a = config['default_delta_a']
            
        target_a = st.number_input("Valeur cible 'a' (Å)", min_value=3.0, max_value=20.0, value=st.session_state.target_a, step=0.05, key='target_a')
        delta_a = st.slider("Marge ±Δa (Å)", 0.01, 2.0, value=st.session_state.delta_a, step=0.01, key='delta_a', help="Ajusté automatiquement au changement de structure.")
        
        st.markdown("---")
        st.subheader("🧪 Filtres Physico-Chimiques & Ordre")
        t_min, t_max = st.slider("Plage Tolérance (t)", 0.80, 1.10, (0.90, 1.05), step=0.01)
        max_delta_chi = st.slider("Δχ max (Stabilité Redox)", 0.0, 3.0, 2.0, step=0.1)
        min_stability = st.slider("Score Stabilité & Ordre Min", 0.0, 1.0, 0.4, step=0.05, help="Combine tolérance, facteur octaédrique et propension à l'ordre B/B'.")
        
        st.markdown("---")
        all_elements = sorted([el for el in ELEMENT_DB.keys() if el != "O"])
        forbidden_elements = st.multiselect("🛑 Exclure Éléments", all_elements, default=[])
        
        st.markdown("---")
        generate_btn = st.button("🚀 LANCER LE CRIBLAGE EXPERT", type="primary", use_container_width=True)

    if generate_btn:
        with st.spinner("⏳ Exécution du moteur vectorisé & calculs d'ordre B/B'..."):
            cn_A = config['cn_A']
            lattice_type = config['lattice']
            r_O = ELEMENT_DB["O"]["radii"][-2][6]
            
            cations_A, cations_B = HTEngine.extract_cations(cn_A, forbidden_elements)
            df_results = HTEngine.vectorized_screening(cations_A, cations_B, r_O, target_a, delta_a, t_min, t_max, max_delta_chi, lattice_type)
            
            df_results = df_results[df_results['stability_score'] >= min_stability]
            df_results = df_results.sort_values(by='stability_score', ascending=False)
            
            st.session_state.df_results = df_results
            st.session_state.config = config

    if "df_results" in st.session_state and not st.session_state.df_results.empty:
        df = st.session_state.df_results
        config = st.session_state.config
        exec_time = st.session_state.get('exec_time', 0.0)
        
        st.success(f"✅ **{len(df)}** combinaisons trouvées en **{exec_time:.4f} secondes** (Analyse d'ordre & géométrie incluses).")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Données & Ordre B/B'", 
            "📈 Visualisations 3D", 
            "💾 Export Fichiers (CIF/POSCAR/QE)", 
            "🤖 Export DeepXDE / ML", 
            "📦 Archive ZIP Complète (DFT)"
        ])
        
        with tab1:
            show_cols = ['Formule', 'Ox_A', 'Ox_B', 'Ox_Bp', 't', 'order_propensity', 'a_calc', 'b_calc', 'c_calc', 'stability_score', 'delta_chi']
            st.dataframe(
                df[show_cols],
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "t": st.column_config.ProgressColumn("Tolérance (t)", min_value=t_min, max_value=t_max, format="%.4f"),
                    "order_propensity": st.column_config.ProgressColumn("Propension Ordre B/B'", min_value=0.0, max_value=1.0, format="%.3f"),
                    "stability_score": st.column_config.ProgressColumn("Score Global", min_value=0.0, max_value=1.0, format="%.3f"),
                    "a_calc": st.column_config.NumberColumn("a (Å)", format="%.4f"),
                    "b_calc": st.column_config.NumberColumn("b (Å)", format="%.4f"),
                    "c_calc": st.column_config.NumberColumn("c (Å)", format="%.4f"),
                }
            )
            
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.scatter(df, x="t", y="stability_score", color="order_propensity", 
                                  title="Stabilité Globale vs Tolérance (Coloré par Ordre B/B')", template="plotly_dark",
                                  hover_data=['Formule', 'a_calc'], color_continuous_scale=px.colors.sequential.Viridis)
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.scatter_3d(df, x="a_calc", y="b_calc", z="c_calc", color="stability_score",
                                     title="Espace des Paramètres de Maille", template="plotly_dark",
                                     hover_data=['Formule'], color_continuous_scale=px.colors.sequential.Inferno)
                st.plotly_chart(fig2, use_container_width=True)
                
        with tab3:
            st.warning("⚠️ Fichiers d'entrée prêts pour DFT (Quantum ESPRESSO `.scf.in`, VASP `POSCAR`, et structure cristalline `.cif`).")
            cols_per_row = 3
            for i in range(0, len(df), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < len(df):
                        row = df.iloc[i+j]
                        with cols[j]:
                            st.markdown(f"**{row['Formule']}**")
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.download_button("CIF", generate_cif(row, config['sg']), f"{row['Formule']}.cif", key=f"cif_{row['Formule']}_{i+j}")
                            with c2:
                                st.download_button("POSCAR", generate_poscar(row), f"POSCAR_{row['Formule']}", key=f"poscar_{row['Formule']}_{i+j}")
                            with c3:
                                st.download_button("QE.in", generate_qe_input(row), f"{row['Formule']}.scf.in", key=f"qe_{row['Formule']}_{i+j}")
                                
        with tab4:
            st.subheader("🤖 Export de Descripteurs pour Machine Learning & DeepXDE (PINNs)")
            st.markdown("Les descripteurs physiques extraits ci-dessous sont formatés pour alimenter des modèles de substitution par apprentissage automatique ou des réseaux de neurones informés par la physique (DeepXDE).")
            
            ml_cols = ['Formule', 't', 'order_propensity', 'a_calc', 'b_calc', 'c_calc', 'beta_calc', 'stability_score', 'delta_chi', 'mean_atomic_mass', 'mean_chi', 'd_e(B)', 'd_e(Bp)']
            df_ml = df[ml_cols]
            
            st.dataframe(df_ml, use_container_width=True, hide_index=True)
            
            csv_data = df_ml.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Télécharger le jeu de données des descripteurs (CSV pour ML / DeepXDE)",
                data=csv_data,
                file_name=f"HTS_A2BBO6_ML_features_{config['sg']}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with tab5:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zf:
                for _, row in df.iterrows():
                    zf.writestr(f"CIF/{row['Formule']}.cif", generate_cif(row, config['sg']))
                    zf.writestr(f"POSCAR/POSCAR_{row['Formule']}", generate_poscar(row))
                    zf.writestr(f"Quantum_ESPRESSO/{row['Formule']}.scf.in", generate_qe_input(row))
                zf.writestr("Screening_Results_and_ML_Features.csv", df.to_csv(index=False))
                
            st.download_button(
                label="⬇️ Télécharger l'archive complète (CIF + POSCAR + QE Inputs + CSV ML)",
                data=zip_buffer.getvalue(),
                file_name=f"HTS_A2BBO6_Complete_Archive_{config['sg']}.zip",
                mime="application/zip",
                use_container_width=True
            )
            
    elif "df_results" in st.session_state and st.session_state.df_results.empty:
        st.error("❌ Aucune combinaison ne satisfait vos critères stricts de stabilité et de géométrie.")
        st.info("💡 **Conseils d'expert :**\n1. Le paramètre de maille par défaut s'ajuste automatiquement. Vérifiez vos filtres.\n2. Baissez le score minimal à 0.2 pour élargir la recherche aux métastables.\n3. Augmentez la marge ±Δa.")

if __name__ == "__main__":
    main()
