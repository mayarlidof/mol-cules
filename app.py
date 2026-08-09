# ==============================================================================
# app.py - Production-Grade HTS A₂BB'O₆ Engine & ML / DeepXDE Toolkit
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import zipfile
import plotly.express as px

# ==============================================================================
# 1. BASE DE DONNÉES CRISTALLOGRAPHIQUE ET ÉLECTRONIQUE (ELEMENTS DB)
# ==============================================================================
# Format des rayons ioniques : {CN: rayon_en_Angstroms, ...}
ELEMENT_DB = {
    "La": {"radii": {6: 1.032, 8: 1.16, 12: 1.36}, "chi": 1.10, "grp": 3, "mass": 138.905},
    "Sr": {"radii": {6: 1.18, 8: 1.26, 12: 1.44}, "chi": 0.95, "grp": 2, "mass": 87.62},
    "Ba": {"radii": {6: 1.35, 8: 1.42, 12: 1.61}, "chi": 0.89, "grp": 2, "mass": 137.33},
    "Ca": {"radii": {6: 1.00, 8: 1.12, 12: 1.34}, "chi": 1.00, "grp": 2, "mass": 40.078},
    "Y":  {"radii": {6: 0.90, 8: 1.015, 12: 1.23}, "chi": 1.22, "grp": 3, "mass": 88.905},
    "Bi": {"radii": {6: 1.17, 8: 1.17, 12: 1.38}, "chi": 2.02, "grp": 5, "mass": 208.98},
    
    # Cations Site B / B' (généralement CN = 6)
    "Ti": {"radii": {6: 0.605}, "chi": 1.54, "grp": 4, "mass": 47.867},
    "Fe": {"radii": {6: 0.645}, "chi": 1.83, "grp": 8, "mass": 55.845},
    "Co": {"radii": {6: 0.65},  "chi": 1.88, "grp": 9, "mass": 58.933},
    "Ni": {"radii": {6: 0.63},  "chi": 1.91, "grp": 10, "mass": 58.693},
    "Mn": {"radii": {6: 0.67},  "chi": 1.55, "grp": 7, "mass": 54.938},
    "Cr": {"radii": {6: 0.615}, "chi": 1.66, "grp": 6, "mass": 51.996},
    "Mo": {"radii": {6: 0.65},  "chi": 2.16, "grp": 6, "mass": 95.95},
    "Re": {"radii": {6: 0.63},  "chi": 1.90, "grp": 7, "mass": 186.21},
    "W":  {"radii": {6: 0.60},  "chi": 2.36, "grp": 6, "mass": 183.84},
    "V":  {"radii": {6: 0.64},  "chi": 1.63, "grp": 5, "mass": 50.942},
    "Sc": {"radii": {6: 0.745}, "chi": 1.36, "grp": 3, "mass": 44.956},
    "Zr": {"radii": {6: 0.72},  "chi": 1.33, "grp": 4, "mass": 91.224},
    "Nb": {"radii": {6: 0.64},  "chi": 1.6,  "grp": 5, "mass": 92.906},
    "Ta": {"radii": {6: 0.64},  "chi": 1.5,  "grp": 5, "mass": 180.95},
    
    # Oxygène (Référence)
    "O":  {"radii": {6: 1.40},  "chi": 3.44, "grp": 6, "mass": 15.999}
}

STRUCTURE_CONFIG = {
    "Double Pérovskite (A₂BB'O₆)": {
        "Cubique (Fm-3m) [SG 225]": {"cn_A": 12, "lattice": "cubic", "sg": "Fm-3m"},
        "Tetragonale (I4/m) [SG 87]": {"cn_A": 12, "lattice": "tetra", "sg": "I4/m"},
        "Monoclinique (P2₁/n) [SG 14]": {"cn_A": 8, "lattice": "mono", "sg": "P21/n"},
        "Orthorhombique (Pbnm) [SG 62]": {"cn_A": 8, "lattice": "ortho", "sg": "Pbnm"},
        "Hexagonal / Trigonal (P-31c)": {"cn_A": 6, "lattice": "hexagonal", "sg": "P-31c"}
    }
}

# ==============================================================================
# 2. MOTEUR DE CRIBLAGE HAUT DÉBIT VECTORISÉ (HTENGINE)
# ==============================================================================
class HTEngine:
    @staticmethod
    def extract_cations(cn_A: int, forbidden_elements: list):
        cations_A = []
        cations_B = []
        
        # États d'oxydation usuels testés pour le criblage
        ox_A_list = [2, 3] if cn_A in [8, 12] else [3]
        ox_B_list = [2, 3, 4, 5]
        
        for el, data in ELEMENT_DB.items():
            if el == "O" or el in forbidden_elements:
                continue
            
            # Extraction Site A
            if el in ["La", "Sr", "Ba", "Ca", "Y", "Bi"]:
                for ox in ox_A_list:
                    if cn_A in data["radii"]:
                        cations_A.append((el, ox, data["radii"][cn_A], data["chi"], data["grp"], data["mass"]))
            
            # Extraction Site B / B' (toujours CN=6 pour les métaux de transition)
            if 6 in data["radii"] and el not in ["La", "Y", "Bi"]:
                for ox in ox_B_list:
                    cations_B.append((el, ox, data["radii"][6], data["chi"], data["grp"], data["mass"]))
                    
        return cations_A, cations_B

    @staticmethod
    def vectorized_screening(cations_A: list, cations_B: list, r_O: float, target_a: float, delta_a: float, t_min: float, t_max: float, max_delta_chi: float, lattice_type: str) -> pd.DataFrame:
        start_time = time.time()
        
        df_A = pd.DataFrame(cations_A, columns=['el_A', 'ox_A', 'r_A', 'chi_A', 'grp_A', 'mass_A'])
        df_B = pd.DataFrame(cations_B, columns=['el_B', 'ox_B', 'r_B', 'chi_B', 'grp_B', 'mass_B'])
        
        df_B['key'] = 1
        df_A_key = df_A.copy()
        df_A_key['key'] = 1
        
        # Combinaison des paires B / B' (B < B')
        df_BxB = pd.merge(df_B, df_B, on='key', suffixes=('', '_Bp'))
        df_BxB = df_BxB[df_BxB['el_B'] < df_BxB['el_Bp']]
        
        # Neutralité de charge globale : 2 * ox_A + ox_B + ox_Bp = 12
        df_BxB['req_2_ox_A'] = 12 - (df_BxB['ox_B'] + df_BxB['ox_Bp'])
        df_BxB = df_BxB[(df_BxB['req_2_ox_A'] > 0) & (df_BxB['req_2_ox_A'] % 2 == 0)]
        df_BxB['req_ox_A'] = df_BxB['req_2_ox_A'] / 2
        
        df_comb = pd.merge(df_BxB, df_A_key, left_on='req_ox_A', right_on='ox_A')
        
        if df_comb.empty:
            st.session_state.exec_time = time.time() - start_time
            return pd.DataFrame()
            
        r_B_eff = (df_comb['r_B'] + df_comb['r_Bp']) / 2.0
        
        # Facteur de tolérance de Goldschmidt (t)
        df_comb['t'] = (df_comb['r_A'] + r_O) / (np.sqrt(2) * (r_B_eff + r_O))
        df_comb = df_comb[(df_comb['t'] >= t_min) & (df_comb['t'] <= t_max)]
        
        # Stabilité Redox (différence d'électronégativité)
        df_comb['delta_chi'] = np.abs(df_comb['chi_B'] - df_comb['chi_Bp'])
        df_comb = df_comb[df_comb['delta_chi'] <= max_delta_chi]
        
        # Propension à l'ordre B/B' (basée sur la différence de taille et de valence)
        delta_r_BBp = np.abs(df_comb['r_B'] - df_comb['r_Bp'])
        delta_z_BBp = np.abs(df_comb['ox_B'] - df_comb['ox_Bp'])
        df_comb['order_propensity'] = (0.5 * (delta_r_BBp / 0.15) + 0.5 * (delta_z_BBp / 2.0)).clip(0.0, 1.0)
        
        # Paramètres de maille effectifs
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
        elif lattice_type in ["ortho", "mono"]:
            df_comb['a_calc'] = df_comb['a_c'] * np.sqrt(2)
            df_comb['b_calc'] = df_comb['a_c'] * 2 if lattice_type == "ortho" else df_comb['a_c']
            df_comb['c_calc'] = df_comb['a_c'] * np.sqrt(2)
            df_comb['beta_calc'] = 90.0 if lattice_type == "ortho" else 135.0
        elif lattice_type == "hexagonal":
            df_comb['a_calc'] = 2 * (r_B_eff + r_O)
            df_comb['b_calc'] = df_comb['a_calc']
            df_comb['c_calc'] = df_comb['a_calc'] * np.sqrt(6)
            df_comb['beta_calc'] = 90.0

        # Filtre sur la cible 'a' et sa marge
        df_comb = df_comb[np.abs(df_comb['a_calc'] - target_a) <= delta_a]
        
        if df_comb.empty:
            st.session_state.exec_time = time.time() - start_time
            return pd.DataFrame()

        mu = r_B_eff / r_O
        score_t = np.exp(-5.0 * np.abs(1.0 - df_comb['t']))
        score_mu = np.exp(-5.0 * np.abs(0.85 - mu))
        df_comb['stability_score'] = (0.7 * (score_t * score_mu) + 0.3 * df_comb['order_propensity']).round(3)
        
        df_comb['Formule'] = df_comb['el_A'] + '2' + df_comb['el_B'] + df_comb['el_Bp'] + 'O6'
        df_comb['d_e(B)'] = np.clip(df_comb['grp_B'] - df_comb['ox_B'], 0, 10)
        df_comb['d_e(Bp)'] = np.clip(df_comb['grp_Bp'] - df_comb['ox_Bp'], 0, 10)
        
        # Descripteurs globaux pour ML / DeepXDE (PINNs)
        df_comb['mean_atomic_mass'] = (2 * df_comb['mass_A'] + df_comb['mass_B'] + df_comb['mass_Bp'] + 6 * ELEMENT_DB['O']['mass']) / 10.0
        df_comb['mean_chi'] = (2 * df_comb['chi_A'] + df_comb['chi_B'] + df_comb['chi_Bp'] + 6 * ELEMENT_DB['O']['chi_pauling']) / 10.0
        
        cols = ['Formule', 'el_A', 'el_B', 'el_Bp', 'ox_A', 'ox_B', 'ox_Bp',
                't', 'a_calc', 'b_calc', 'c_calc', 'beta_calc', 'stability_score', 
                'order_propensity', 'delta_chi', 'd_e(B)', 'd_e(Bp)', 'mean_atomic_mass', 'mean_chi']
        
        df_final = df_comb[cols].copy()
        df_final.rename(columns={'ox_A': 'Ox_A', 'ox_B': 'Ox_B', 'ox_Bp': 'Ox_Bp'}, inplace=True)
        
        st.session_state.exec_time = time.time() - start_time
        return df_final.reset_index(drop=True)

# ==============================================================================
# 3. GÉNÉRATEURS DE FICHIERS DE STRUCTURE (CIF, POSCAR, QUANTUM ESPRESSO)
# ==============================================================================
def generate_cif(row, space_group):
    return f"""data_{row['Formule']}
_symmetry_space_group_name_H-M   '{space_group}'
_cell_length_a                   {row['a_calc']:.5f}
_cell_length_b                   {row['b_calc']:.5f}
_cell_length_c                   {row['c_calc']:.5f}
_cell_angle_alpha                90.00000
_cell_angle_beta                 {row['beta_calc']:.5f}
_cell_angle_gamma                90.00000

loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
{row['el_A']}1  0.25000  0.25000  0.25000  1.0
{row['el_A']}2  0.75000  0.75000  0.75000  1.0
{row['el_B']}   0.00000  0.00000  0.00000  1.0
{row['el_Bp']}  0.50000  0.50000  0.50000  1.0
O1     0.25000  0.00000  0.00000  1.0
O2     0.00000  0.25000  0.00000  1.0
O3     0.00000  0.00000  0.25000  1.0
O4     0.75000  0.50000  0.50000  1.0
O5     0.50000  0.75000  0.50000  1.0
O6     0.50000  0.50000  0.75000  1.0
"""

def generate_poscar(row):
    return f"""{row['Formule']} generated by HTS A2BBO6 Engine
1.0
{row['a_calc']:.6f} 0.000000 0.000000
0.000000 {row['b_calc']:.6f} 0.000000
0.000000 0.000000 {row['c_calc']:.6f}
{row['el_A']} {row['el_B']} {row['el_Bp']} O
2 1 1 6
Direct
0.25 0.25 0.25
0.75 0.75 0.75
0.00 0.00 0.00
0.50 0.50 0.50
0.25 0.00 0.00
0.00 0.25 0.00
0.00 0.00 0.25
0.75 0.50 0.50
0.50 0.75 0.50
0.50 0.50 0.75
"""

def generate_qe_input(row):
    return f"""&control
  calculation = 'scf'
  restart_mode = 'from_scratch',
  prefix = '{row['Formule']}',
  pseudo_dir = './',
  outdir = './out/'
/
&system
  ibrav = 0, nat = 10, ntyp = 4,
  ecutwfc = 50.0, ecutrho = 400.0,
  occupations = 'smearing', smearing = 'cold', degauss = 0.02
/
&electrons
  conv_thr = 1.0d-8,
  mixing_beta = 0.4
/
ATOMIC_SPECIES
{row['el_A']}  {ELEMENT_DB[row['el_A']]['mass']}  {row['el_A']}.rel-pbe-rrkjus.UPF
{row['el_B']}  {ELEMENT_DB[row['el_B']]['mass']}  {row['el_B']}.rel-pbe-rrkjus.UPF
{row['el_Bp']} {ELEMENT_DB[row['el_Bp']]['mass']} {row['el_Bp']}.rel-pbe-rrkjus.UPF
O     15.999  O.rel-pbe-rrkjus.UPF

CELL_PARAMETERS (angstrom)
 {row['a_calc']:.6f} 0.000000 0.000000
 0.000000 {row['b_calc']:.6f} 0.000000
 0.000000 0.000000 {row['c_calc']:.6f}

ATOMIC_POSITIONS (crystal)
{row['el_A']}  0.25000  0.25000  0.25000
{row['el_A']}  0.75000  0.75000  0.75000
{row['el_B']}   0.00000  0.00000  0.00000
{row['el_Bp']}  0.50000  0.50000  0.50000
O     0.25000  0.00000  0.00000
O     0.00000  0.25000  0.00000
O     0.00000  0.00000  0.25000
O     0.75000  0.50000  0.50000
O     0.50000  0.75000  0.50000
O     0.50000  0.50000  0.75000
K_POINTS automatic
4 4 4 1 1 0
"""

# ==============================================================================
# 4. INTERFACE UTILISATEUR (STREAMLIT UI/UX)
# ==============================================================================
def main():
    st.set_page_config(page_title="⚛️ HTS A2BB'O6 Advanced Engine", layout="wide", initial_sidebar_state="expanded")
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
        
        # --- AUTOMATISATION INITIALE INTELLIGENTE DE LA CIBLE 'a' ---
        cn_A_temp = config['cn_A']
        r_O_temp = ELEMENT_DB["O"]["radii"][-2][6]
        
        if 'last_struct_family' not in st.session_state or st.session_state.last_struct_family != struct_family:
            st.session_state.last_struct_family = struct_family
            try:
                temp_cations_A, temp_cations_B = HTEngine.extract_cations(cn_A_temp, [])
                if temp_cations_A and temp_cations_B:
                    mean_rA = np.mean([c[2] for c in temp_cations_A])
                    mean_rB = np.mean([c[2] for c in temp_cations_B])
                    if config['lattice'] == 'cubic':
                        st.session_state.target_a_val = float(np.round(np.sqrt(2) * (mean_rA + r_O_temp) + 2 * (mean_rB + r_O_temp) * np.sqrt(2), 2))
                    elif config['lattice'] in ['tetra', 'mono', 'ortho']:
                        st.session_state.target_a_val = float(np.round(2.0 * (mean_rA + r_O_temp), 2))
                    elif config['lattice'] == 'hexagonal':
                        st.session_state.target_a_val = float(np.round(2.0 * (mean_rB + r_O_temp), 2))
                    else:
                        st.session_state.target_a_val = 7.80
                else:
                    st.session_state.target_a_val = 7.80
            except Exception:
                st.session_state.target_a_val = 7.80

        target_a = st.number_input("Valeur cible 'a' (Å)", min_value=3.0, max_value=20.0, 
                                   value=float(st.session_state.get('target_a_val', 7.80)), step=0.05,
                                   key="target_a_input")
        delta_a = st.slider("Marge ±Δa (Å)", 0.01, 5.0, 2.00, step=0.05)
        
        st.markdown("---")
        st.subheader("🧪 Filtres Physico-Chimiques & Ordre")
        t_min, t_max = st.slider("Plage Tolérance (t)", 0.80, 1.10, (0.80, 1.10), step=0.01)
        max_delta_chi = st.slider("Δχ max (Stabilité Redox)", 0.0, 3.0, 2.5, step=0.1)
        min_stability = st.slider("Score Stabilité & Ordre Min", 0.0, 1.0, 0.2, step=0.05, help="Seuil abaissé pour capturer les phases métastables.")
        
        st.markdown("---")
        all_elements = sorted([el for el in ELEMENT_DB.keys() if el != "O"])
        default_forbidden = [el for el in ["Li", "Na", "K", "Rb", "Cs"] if el in all_elements]
        forbidden_elements = st.multiselect("🛑 Exclure Éléments", all_elements, default=default_forbidden)
        
        st.markdown("---")
        generate_btn = st.button("🚀 LANCER LE CRIBLAGE EXPERT", type="primary", use_container_width=True)

    if generate_btn:
        with st.spinner("⏳ Exécution du moteur vectorisé & calculs d'ordre B/B'..."):
            cn_A = config['cn_A']
            lattice_type = config['lattice']
            r_O = ELEMENT_DB["O"]["radii"][-2][6]
            
            cations_A, cations_B = HTEngine.extract_cations(cn_A, forbidden_elements)
            df_results = HTEngine.vectorized_screening(cations_A, cations_B, r_O, target_a, delta_a, t_min, t_max, max_delta_chi, lattice_type)
            
            if not df_results.empty:
                df_results = df_results[df_results['stability_score'] >= min_stability]
                df_results = df_results.sort_values(by='stability_score', ascending=False)
            
            st.session_state.df_results = df_results
            st.session_state.config = config

    # Gestion de l'affichage des résultats
    if "df_results" in st.session_state:
        df = st.session_state.df_results
        config = st.session_state.config
        exec_time = st.session_state.get('exec_time', 0.0)
        
        if df.empty:
            st.error("❌ Aucune combinaison ne satisfait vos critères stricts de stabilité et de géométrie.")
            st.info("💡 **Conseils d'expert :**\n1. Baissez le score minimal ou élargissez la marge ±Δa.\n2. Utilisez le module d'optimisation de 'a' ci-dessous si vous avez des résultats en mémoire.")
        else:
            st.success(f"✅ **{len(df)}** combinaisons trouvées en **{exec_time:.4f} secondes** (Analyse d'ordre & géométrie incluses).")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Données & Optimiseur 'a'", 
                "📈 Visualisations 3D", 
                "💾 Export Fichiers (CIF/POSCAR/QE)", 
                "🤖 Export DeepXDE / ML", 
                "📦 Archive ZIP Complète (DFT)"
            ])
            
            with tab1:
                st.subheader("📊 Données & Analyse d'Ordre B/B'")
                
                # ==============================================================
                # ✨ MODULE AVANCÉ : OPTIMISEUR ROBUSTE DE LA CIBLE 'a' PAR SÉLECTION
                # ==============================================================
                st.markdown("---")
                st.markdown("### 🎯 Module d'Optimisation Interactive du Paramètre de Maille `#a#`")
                st.info("Sélectionnez une structure de référence dans la liste ci-dessous pour extraire son paramètre de maille `a_calc`, redéfinir la cible optimale et explorer instantanément de nouvelles combinaisons isomorphes ou métastables.")
                
                col_opt1, col_opt2 = st.columns([3, 1])
                with col_opt1:
                    selected_formula_for_opt = st.selectbox(
                        "Structure de référence pour l'optimisation :",
                        df['Formule'].tolist(),
                        key="select_formula_opt"
                    )
                with col_opt2:
                    st.write("") 
                    st.write("")
                    btn_optimize_from_struct = st.button("✨ Optimiser sur ce 'a'", use_container_width=True, type="secondary")
                
                if btn_optimize_from_struct:
                    target_row = df[df['Formule'] == selected_formula_for_opt].iloc[0]
                    optimal_a_target = float(target_row['a_calc'])
                    
                    st.session_state.target_a_val = optimal_a_target
                    
                    with st.spinner(f"🔄 Relancement du criblage autour de a = {optimal_a_target:.3f} Å (recherche de structures alternatives)..."):
                        cn_A = config['cn_A']
                        lattice_type = config['lattice']
                        r_O = ELEMENT_DB["O"]["radii"][-2][6]
                        
                        cations_A, cations_B = HTEngine.extract_cations(cn_A, forbidden_elements if 'forbidden_elements' in locals() else [])
                        df_new_results = HTEngine.vectorized_screening(
                            cations_A, cations_B, r_O, optimal_a_target, 2.00, t_min, t_max, max_delta_chi, lattice_type
                        )
                        
                        if not df_new_results.empty:
                            df_new_results = df_new_results[df_new_results['stability_score'] >= min_stability]
                            df_new_results = df_new_results.sort_values(by='stability_score', ascending=False)
                        
                        st.session_state.df_results = df_new_results
                        st.success(f"🎯 Cible 'a' optimisée à {optimal_a_target:.3f} Å ! {len(df_new_results)} combinaisons alternatives trouvées.")
                        st.rerun()

                st.markdown("---")
                
                # Tableau principal des résultats
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

if __name__ == "__main__":
    main()
