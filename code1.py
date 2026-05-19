import numpy as np

from ex1 import (
    load_data,
    run_minimization,
    uniform_disk_visibility2
)

def run_analysis(data_root, star_name, date, tplstart, bases):

    # =========================
    # LOAD DATA
    # =========================

    data = load_data(
        data_root=data_root,
        star=star_name,
        date=date,
        bases=bases,
        tplstart=tplstart,
    )

    # =========================
    # RUN MINIMIZATION
    # =========================

    result = run_minimization(data)

    # =========================
    # GLOBAL RESULTS
    # =========================

    diam_global = result["diam_global"]

    diam_init = result["diam_candidates"]

    valid = result["valid_bases"]

    diam = diam_init[:, :, 0]

    baseline_diam = diam.reshape(-1)

    valid_flat = valid.reshape(-1)

    diam_per_baseline = baseline_diam[valid_flat]

    diam_per_baseline = diam_per_baseline[
        (~np.isnan(diam_per_baseline))
        & (diam_per_baseline > 0)
    ]

    if len(diam_per_baseline) == 0:
        diam_per_baseline = np.array([diam_global])

    # =========================
    # LABELS
    # =========================

    baseline_labels = []

    for state, base_pair in zip(
        data["bcd_states"],
        data["base"]
    ):

        for baseline in base_pair:

            baseline_labels.append(
                f"{state}:{baseline[0]}-{baseline[1]}"
            )

    # =========================
    # TABLE RESULTS
    # =========================

    baseline_results = []

    for label, d, is_valid in zip(
        baseline_labels,
        baseline_diam,
        valid_flat,
    ):

        baseline_results.append({

            "baseline": label,

            "diam": (
                float(d)
                if not np.isnan(d)
                else None
            ),

            "valid": bool(is_valid),

        })

    # ============================================================
    # TRUE PHYSICAL TRANSFER FUNCTIONS
    # FT = V²_mes / V²_ideal
    # ============================================================

    

    wavelength_um = result["lambda_ref"]
    spatial_frq = result["freq_cube"]

    # IMPORTANT : données V² brutes, avant toute normalisation
    vis2_initial_cube = data["vis2_initial_cube"]

    diam_fit = result["diam_global"]

    NB_BCD, nrow, nwave = spatial_frq.shape

    transfer_functions = []

    counter = 0

    for ii in range(NB_BCD):

        for i in range(nrow):

            # =========================================
            # V² idéal disque uniforme brut
            # =========================================
            vis2_ideal = uniform_disk_visibility2(
                spatial_frq[ii, i, :],
                diam_fit
            )

          
            # =========================================
            # FT = V²mes / V²ideal
            # =========================================
            tf_mes = np.sqrt(
                vis2_initial_cube[ii, i]
                / vis2_ideal
            )

            transfer_functions.append({
                "baseline": baseline_labels[counter],
                "wavelength_um": wavelength_um.tolist(),
                "tf_curve": tf_mes.tolist(),
                "valid": bool(valid[ii, i]),
            })

            counter += 1
    # =========================
    # RETURN
    # =========================

    return {

        "diam_catalog": float(data["diam1"]),

        "diam_catalog_err": float(data["emas"]),

        "diam_global": float(diam_global),

        "chi2_global": float(result["chi2_global"]),

        "baseline_results": baseline_results,

        "n_valid_baselines": int(len(diam_per_baseline)),

        "diam_median": float(np.median(diam_per_baseline)),

        "diam_mean": float(np.mean(diam_per_baseline)),

        "diam_std": float(np.std(diam_per_baseline)),

        "transfer_functions": transfer_functions,
    }