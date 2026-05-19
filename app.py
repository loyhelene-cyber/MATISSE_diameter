# app.py
import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from code1 import run_analysis

st.set_page_config(
    page_title="MATISSE - Calibrator angular diameter",
    layout="wide"
)

# =========================
# STYLE PERSONNALISÉ
# =========================
st.markdown("""
<style>

/* =========================
   GLOBAL BACKGROUND
========================= */
.stApp {
    background: linear-gradient(135deg, #EAF6F8 0%, #F4E6C1 100%);
    color: #0B2D42;
    font-family: 'Segoe UI', sans-serif;
}

/* =========================
   TITLES
========================= */
h1 {
    color: #0B2D42;
    font-weight: 700;
}

h2, h3 {
    color: #1F6F8B;
}

/* =========================
   SIDEBAR
========================= */
[data-testid="stSidebar"] {
    background-color: #0B2D42;
    padding-top: 1rem;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label {
    color: white !important;
}

/* Help icon in sidebar */
[data-testid="stSidebar"] button svg {
    fill: white !important;
}

/* =========================
   INPUT FIELDS
========================= */
[data-baseweb="input"] {
    background-color: white !important;
    border-radius: 10px !important;
    border: none !important;
}

[data-baseweb="input"] input {
    color: #0B2D42 !important;
    font-weight: 500;
}

/* Placeholder */
::placeholder {
    color: #888 !important;
}

/* =========================
   BUTTONS
========================= */
.stButton > button {
    background-color: #1F6F8B;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.7rem 1.2rem;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}

.stButton > button:hover {
    background-color: #0B2D42;
    transform: scale(1.03);
}

/* =========================
   METRICS
========================= */
[data-testid="stMetric"] {
    background-color: rgba(255,255,255,0.85);
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid #D8C08A;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}

/* =========================
   TABLES
========================= */
div[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 12px;
    padding: 0.5rem;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

/* =========================
   SUCCESS / INFO BOX
========================= */
.stAlert {
    border-radius: 12px;
}

/* =========================
   BACK BUTTON
========================= */
button[kind="secondary"] {
    background-color: #F4E6C1 !important;
    color: #0B2D42 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================
# NAVIGATION
# =========================
if "page" not in st.session_state:
    st.session_state.page = "intro"


# =============================================================================
# INTRO
# =============================================================================
if st.session_state.page == "intro":

    st.title("Estimation of the angular diameter of MATISSE calibrators")
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image("banniere.png", width=800)
    st.markdown("""
## MATISSE Calibrator Angular Diameter Estimation

This application provides an automated pipeline to estimate the **angular diameter of interferometric calibrators**
observed with the MATISSE instrument (VLTI), based on OIFITS data. For more details on the method and results, please refer to the following publication:
[Robbe-Dubois et al. — MATISSE calibrators](https://hal.science/hal-03473191v2/file/MATISSE_Calibrator.pdf)

### Methodology

The estimation relies on a multi-step procedure:

- **Visibility normalization** using a reference baseline and wavelength;
- **Outlier rejection** based on residual analysis;
- **Transfer function modelling** to account for instrumental effects;
- **Simultaneous fitting** of the transfer function and the stellar diameter;
- **Global diameter estimation** from all valid baselines.

### Outputs

The application provides:

- The **catalog diameter** of the calibrator;
- The **fitted global angular diameter**;
- The **χ² indicator**;
- A set of **baseline-by-baseline diameter estimates** (advanced diagnostic);

### Notes

- The method is currently implemented 3.5 µm ± 0.25 µm (L band) because it is the most stable region for MATISSE data in terms of signal-to-noise ratio and transfer function constraints.
- The N-band pipeline is not implemented yet.
- The uncertainties on the fitted diameter are not computed in this version. 

### Please select the spectral band to start the analysis
""")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Analyse L band", type="primary", use_container_width=True):
            st.session_state.page = "L"
            st.rerun()

    with col2:
        if st.button("Analyse N band", use_container_width=True):
            st.session_state.page = "N"
            st.rerun()


# =============================================================================
# L BAND
# =============================================================================
elif st.session_state.page == "L":

    if st.button("← Back"):
        st.session_state.page = "intro"
        st.rerun()

    st.title("L band — Calibrator angular diameter")

    st.markdown("""
    The analysis is performed in the **L band**.

    The selected wavelength range is:

    **3.5 µm ± 0.25 µm** (≈ 3.25 – 3.75 µm)

    This range is used because it is the most stable region for MATISSE data
    in terms of signal-to-noise ratio. It is also the range where the transfer
    function is best constrained, allowing a more accurate estimation of the
    angular diameter.
    """)

    with st.sidebar:
        st.header("Observation parameters")

        data_root = st.text_input(
            "Root folder of the data",
            value="data",
            help="Path to the folder containing the MATISSE data files"
        )

        star_name = st.text_input("Calibrator name", value="alfCha")
        date = st.text_input("Observation date", value="2021-03-27")
        tplstart = st.text_input("TPL START", value="033203")
        bases = st.text_input("Bases", value="U1U2U3U4")

        run_button = st.button("Run analysis", type="primary")

    if run_button:
        try:
            with st.spinner("Running analysis..."):
                results = run_analysis(
                    data_root=Path(data_root),
                    star_name=star_name,
                    date=date,
                    tplstart=tplstart,
                    bases=bases
                )

            st.success("Analysis completed.")

            col1, col2, col3 = st.columns(3)

            col1.metric("Catalog diameter", f"{results['diam_catalog']:.3f} mas")
            col1.caption(f"Uncertainty: {results['diam_catalog_err']:.3f} mas")

            col2.metric("Optimized diameter", f"{results['diam_global']:.3f} mas")
            col2.caption("Although not computed here, the method typically yields uncertainties of ~1–4% (see reference).")

            col3.metric("Global χ²", f"{results['chi2_global']:.3f}")

            st.markdown("""The detailed optimization results are intended for advanced users.  
            They provide angular diameter estimates for each baseline.  
             **Reference:**  
            [Robbe-Dubois et al. — MATISSE calibrators](https://hal.science/hal-03473191v2/file/MATISSE_Calibrator.pdf)""")

            with st.expander("Afficher les fonctions de transfert par baseline"):
                st.markdown(
                    "Cliquez pour ouvrir la fenêtre détaillée des fonctions de transfert. "
                    "Chaque plot correspond à une baseline et montre si elle est retenue ou rejetée."
                )
                
            tf_items = results["transfer_functions"]

            bcd_order = ["IN/IN", "IN/OUT", "OUT/IN", "OUT/OUT"]

            def get_bcd(tf_item):
                txt = tf_item.get("bcd", tf_item.get("baseline", ""))

                if "IN_IN" in txt or "IN/IN" in txt:
                    return "IN/IN"
                if "IN_OUT" in txt or "IN/OUT" in txt:
                    return "IN/OUT"
                if "OUT_IN" in txt or "OUT/IN" in txt:
                    return "OUT/IN"
                if "OUT_OUT" in txt or "OUT/OUT" in txt:
                    return "OUT/OUT"

                return None

            tf_by_bcd = {bcd: [] for bcd in bcd_order}

            for item in tf_items:
                bcd = get_bcd(item)
                if bcd is not None:
                    tf_by_bcd[bcd].append(item)

            # Titres des colonnes
            cols = st.columns(4)
            for col, bcd in zip(cols, bcd_order):
                with col:
                    st.markdown(f"### {bcd}")

            # 6 lignes × 4 colonnes
            for row in range(6):
                cols = st.columns(4)

                for col_idx, bcd in enumerate(bcd_order):
                    with cols[col_idx]:
                        items = tf_by_bcd[bcd]

                        if row >= len(items):
                            st.empty()
                            continue

                        tf_item = items[row]

                        fig, ax = plt.subplots(figsize=(3.8, 2.8))

                        curve = tf_item["tf_curve"]
                        wave = tf_item["wavelength_um"]

                        if tf_item["valid"]:
                            ax.plot(wave, curve, color="#0B2D42", linewidth=1.5)
                        else:
                            ax.plot(wave, curve, linestyle="--", color="#A83232", linewidth=1.5)

                        title = tf_item["baseline"]
                        if not tf_item["valid"]:
                            title += " (rejected)"

                        ax.set_title(title, fontsize=9)
                        ax.set_xlabel("λ (µm)", fontsize=8)
                        ax.set_ylabel("TF", fontsize=8)
                        ax.tick_params(axis="both", labelsize=7)
                        ax.grid(True, alpha=0.25)
                        ax.set_ylim(0.4, 0.8)

                        st.pyplot(fig)
                        plt.close(fig)
            with st.expander("Show detailed optimization results by baseline"):
                df = pd.DataFrame(results["baseline_results"])
                st.dataframe(df, use_container_width=True)

            st.subheader("Summary")
            st.write(f"Baselines taken into account: {results['n_valid_baselines']}")
            #st.write(f"Diameter median: {results['diam_median']:.3f} mas")
           # st.write(f"Diameter mean: {results['diam_mean']:.3f} mas")
           # st.write(f"Standard deviation: {results['diam_std']:.3f} mas")

        except FileNotFoundError as e:
            st.error("File not found")
            st.code(str(e))

        except Exception as e:
            st.error("Error during analysis")
            st.exception(e)

    else:
        st.info("Fill parameters and click on Run analysis.")


# =============================================================================
# N BAND
# =============================================================================
elif st.session_state.page == "N":

    if st.button("← Back"):
        st.session_state.page = "intro"
        st.rerun()

    st.title("N band — MATISSE")

    st.warning("Method not available yet.")

    st.markdown("""
    The N-band pipeline is not implemented yet.

    Please use the L-band analysis.
    """)
