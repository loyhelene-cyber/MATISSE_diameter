# -*- coding: utf-8 -*-

"""
Estimation of the angular diameter of interferométric calibrators from MATISSE OIFITS data.

The model :
    
    - Transfer Function (FT) spectral model :
         TF(λ) = 1 + b(λ − λ_ref) + c(λ − λ_ref)²
         
    -Measured squared visibility :
         V²_ref = TF × [UD(λ)/UD(λ_ref)] / [UD(λ)/UD(λ_ref)]_short_baseline
where:
- UD = uniform disk visibility model
- λ_ref = reference wavelength
- short_baseline = shortest baseline used for normalization

Author: loyhe
Created: Jan 2026
    
"""
#===== LIBRARY & IMPORTS =====
import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
from scipy.special import jv
from scipy.optimize import minimize
from pathlib import Path

#==== MAIN MATISSE'S PARAMETERS  ====

#Number of Beam Commutation Device States 
NB_BCD = 4

#Number of physical baselines in MATISSE (4 telescopes -> 6 baselines)
N_BASELINES = 6

# Spectral stability window of MATISSE around 3.5 µm
LAMBDA_CENTER = 3.5
LAMBDA_HALF_WIDTH  = 0.25

#==== CONSTANTS ===
kMAD = 3.5 

#==== FUNCTIONS DEFINITION  ====


def extract_baselines(oifits_file):
    """
    Extracts the baseline names corresponding to the squared visibility (v²) measurements.

        The function retrieves the list of telescopes and their indices from the OIFITS file and returns the names of the 6 baselines composing MATISSE for the v² visibility measurements.

        Arguments:

            oifits_file (str): path to the OIFITS file to process.

        Returns:
            list: List of telescope names for the 6 baselines (12 elements).
    """


    with fits.open(oifits_file) as hdul :  #on ouvre le fichier fits
    
        #on récupère la liste des télescopes 
        sta_name=hdul['OI_ARRAY'].data['STA_NAME'] #on récupère la colonne STA_NAME de la table OI_ARRAYS du fichier OIFITS et on stocke dans sta_name
        sta_index=hdul['OI_ARRAY'].data['STA_INDEX'] #on récupère la colonne STA_INDEX de la table OI_ARRAYS du fichier OIFITS et on stocke dans sta_index
        base_station_indices = np.array(hdul['TF2'].data['STA_INDEX'][0:6] ) 
       
        index_and_name = dict(zip(sta_index, sta_name)) #on crée un dictionnaire qui associe chaque numéro de télescope à son nom
        
        baselines_names = [((index_and_name[tel1]),(index_and_name[tel2]))
               
            
        for tel1, tel2 in base_station_indices]
    
        return baselines_names 

def uniform_disk_visibility2(spatial_frq, diameter):
    """
    Squared visibility of a uniform disk.

    Parameters
    ----------
    spatial_frq : array_like
        Spatial frequency (rad^-1)

    diameter : float
        Angular diameter in mas

    Returns
    -------
    vis2_model : array_like
        Squared visibility
    """

    # mas -> rad
    diameter_rad = diameter * 1e-3 * np.pi / (180.0 * 3600.0)

    z = np.pi * spatial_frq * diameter_rad

    vis2_model = np.ones_like(z, dtype=float)

    mask = np.abs(z) > 1e-12

    vis2_model[mask] = (
        2.0 * jv(1, z[mask]) / z[mask]
    ) ** 2

    return vis2_model

def ud_ratio(spatial_frq, spatial_frq_bc, spatial_frq_ref, spatial_frq_ref_bc, diameter):
    """
    Computes the normalized squared visibility ratio of the uniform disk,
    using the short baseline and the reference wavelength.
    """
    return (
        (uniform_disk_visibility2(spatial_frq, diameter) / uniform_disk_visibility2(spatial_frq_ref, diameter))
        / (uniform_disk_visibility2(spatial_frq_bc, diameter) / uniform_disk_visibility2(spatial_frq_ref_bc, diameter))
    )

def transfer_function(delta_lambda, b, c):
    """
    Computes the transfer function TF = 1 + b*Δλ + c*Δλ²,
    where Δλ = λ - λ_ref.
    """
    return 1.0 + b * delta_lambda + c * delta_lambda**2

def objective_function_transfer_first_minimization(parameters1, spatial_frq, spatial_frq_bc,spatial_frq_ref,spatial_frq_ref_bc,vis2_ref,vis2_err,delta_lambda,weight,diameter_vizier):
    """
    This function computes the chi-square (χ²) for the first minimization with a fixed diameter.
    The parameters b and c of the transfer function are minimized.

    Arguments:

        parameters1 (tuple): parameters (b, c) of the transfer function.
        spatial_frq (array): spatial frequency for the considered baseline.
        spatial_frq_bc (array): spatial frequency for the shortest baseline.
        spatial_frq_ref (array): reference spatial frequency for the considered baseline.
        spatial_frq_ref_bc (array): reference spatial frequency for the shortest baseline.
        vis2_ref (array): observed reference squared visibility.
        vis2_err (array): associated uncertainty.
        delta_lambda (array): wavelength shift.
        weight (array): weight of each measurement.
        diameter_vizier (float): fixed uniform disk diameter from the JSDC astronomical catalog.

    Returns:
        float: minimized chi-square (χ²) value.
        
        """
    b,c = parameters1
    
    ud_model = ud_ratio(spatial_frq,spatial_frq_bc,spatial_frq_ref,spatial_frq_ref_bc,diameter_vizier)
    tf = transfer_function(delta_lambda, b, c)
    chi2_terms = weight * ((vis2_ref - tf * ud_model) / vis2_err) ** 2
    return float(np.sum(chi2_terms))

def objective_function_transfer_and_diameter_second_minimization(parameters2,spatial_frq,spatial_frq_bc,spatial_frq_ref,spatial_frq_ref_bc,vis2_ref,vis2_err,delta_lambda,weight):
    """
    This function computes the chi-square (χ²) for the second minimization with both the diameter and the transfer function parameters as free variables.
    The parameters b and c of the transfer function, as well as the diameter, are minimized simultaneously.

    Arguments:

        parameters2 (tuple): parameters (b, c, diameter to be fitted).
        spatial_frq (array): spatial frequency for the considered baseline.
        spatial_frq_bc (array): spatial frequency for the shortest baseline.
        spatial_frq_ref (array): reference spatial frequency for the considered baseline.
        spatial_frq_ref_bc (array): reference spatial frequency for the shortest baseline.
        vis2_ref (array): observed reference squared visibility.
        vis2_err (array): associated uncertainty.
        delta_lambda (array): wavelength shift.
        weight (array): weight of each measurement.

    Returns:
        float: chi-square (χ²) value.
           """
    b, c, diameter = parameters2
    ud_model = ud_ratio(spatial_frq, spatial_frq_bc, spatial_frq_ref, spatial_frq_ref_bc, diameter)
    tf = transfer_function(delta_lambda, b, c)
    chi2_terms = weight * ((vis2_ref - tf * ud_model) / vis2_err) ** 2
    return float(np.sum(chi2_terms))

def objective_function_transfer_bc_fixed_third_minimization(parameters3, spatial_frq, spatial_frq_bc,spatial_frq_ref, spatial_frq_ref_bc,vis2_ref, vis2_err, delta_lambda, weight,b_fixed, c_fixed):
  """
  
   This function computes the chi-square (χ²) for the third minimization with the diameter and transfer function parameters fixed.
   Only the diameter is minimized, while b and c are fixed to the previously determined values.

   Arguments:

       parameters3 (tuple): parameter (diameter to be fitted).
       spatial_frq (array): spatial frequency for the considered baseline.
       spatial_frq_bc (array): spatial frequency for the shortest baseline.
       spatial_frq_ref (array): reference spatial frequency for the considered baseline.
       spatial_frq_ref_bc (array): reference spatial frequency for the shortest baseline.
       vis2_ref (array): observed reference squared visibility.
       vis2_err (array): associated uncertainty.
       delta_lambda (array): wavelength shift.
       weight (array): weight of each measurement.
       b_fixed (float): fixed transfer function parameter.
       c_fixed (float): fixed transfer function parameter.

    Returns:
        float: chi-square (χ²) value.
  """
  diameter = parameters3[0]

  ud_model = ud_ratio(spatial_frq,spatial_frq_bc,spatial_frq_ref,spatial_frq_ref_bc,diameter)

  tf = transfer_function(delta_lambda, b_fixed, c_fixed)

  chi2_terms = weight * ((vis2_ref - tf * ud_model) / vis2_err) ** 2
  return float(np.sum(chi2_terms))

def objective_function_transfer_fourth_minimization(parameters4, spatial_frq, spatial_frq_bc,spatial_frq_ref, spatial_frq_ref_bc,vis2_ref, vis2_err, delta_lambda, weight,diameter):
    """
    
    This function computes the chi-square (χ²) for the fourth and final minimization step of the process.
    The parameters b and c are adjusted while keeping the diameter fixed (result from the previous steps).

    Arguments:

        parameters4 (tuple): parameters (b and c to be fitted).
        spatial_frq (array): spatial frequency for the considered baseline.
        spatial_frq_bc (array): spatial frequency for the shortest baseline.
        spatial_frq_ref (array): reference spatial frequency for the considered baseline.
        spatial_frq_ref_bc (array): reference spatial frequency for the shortest baseline.
        vis2_ref (array): observed reference squared visibility.
        vis2_err (array): uncertainty associated with the measurements.
        delta_lambda (array): wavelength shift.
        weight (array): weight of each measurement.
        
    
    """
    b, c = parameters4
    ud_model = ud_ratio(spatial_frq, spatial_frq_bc, spatial_frq_ref, spatial_frq_ref_bc, diameter)
    tf = transfer_function(delta_lambda, b, c)

    chi2_terms = weight * ((vis2_ref - tf * ud_model) / vis2_err) ** 2
    return float(np.sum(chi2_terms))

def tri_res(kMAD,spatial_frq, spatial_frq_bc,spatial_frq_ref, spatial_frq_ref_bc,vis2_ref, vis2_err, delta_lambda, weight, b, c, diameter, i_ref_wave, bs):
    
    """
    Outlier rejection via asymmetric MAD on normalized residuals.
    Sets the weights of outlier measurements to zero.
    """

    #  modèle 
    ud = ud_ratio(spatial_frq, spatial_frq_bc, spatial_frq_ref, spatial_frq_ref_bc, diameter)
    tf = transfer_function(delta_lambda, b,c)
    vis2_model = tf * ud

    # résidus normalisés
    resid = (vis2_ref - vis2_model) / vis2_err

    # résidus utilisés pour calculer la MAD : 
    NB_BCD, nrow, nwave = resid.shape

    wave_mask = np.ones(nwave, dtype=bool)
    wave_mask[i_ref_wave] = False

    subset_list = []
    for ii in range(NB_BCD):
        j = int(bs[ii, 0, 0])
        row_mask = np.ones(nrow, dtype=bool)
        row_mask[j] = False
        subset_list.append(resid[ii, row_mask, :][:, wave_mask])

    subset = np.array(subset_list)

    #  médiane + MAD asymétrique
    med = np.median(subset)
    low = subset[subset < med]
    up = subset[subset > med]

    LOWMAD = np.median(np.abs(low - med)) if low.size else 0.0
    UPMAD = np.median(np.abs(up - med)) if up.size else 0.0


    MAD_min = med - kMAD * 1.4826 * LOWMAD
    MAD_max = med + kMAD * 1.4826 * UPMAD

    # outliers -> poids à zéro
    mask = (resid < MAD_min) | (resid > MAD_max)
    weight[mask] = 0.0

    # exclusions systématiques
    weight[:, :, i_ref_wave] = 0.0
 
    for ii in range(resid.shape[0]):  # NB_BCD
        j = bs[ii, 0, 0]
        weight[ii, j, :] = 0.0
 
    return {"weight": weight, "residual": resid}

def load_data(data_root, star, date, bases, tplstart):

    starname = []

    prefix = f"{date}T{tplstart}_{star}_{bases}_IR-LM_LOW"
    file0 = data_root / prefix

    bcd_states = ["IN_IN", "OUT_OUT", "IN_OUT", "OUT_IN"]

    base = np.empty((NB_BCD, N_BASELINES, 2), dtype="U6")

    diam1 = None
    emas = None

    vis2_cube = None
    errvis2_cube = None
    freq_cube = None
    lambda_cube_ref = None

    for idx, state in enumerate(bcd_states):

        nom = str(file0) + f"_{state}_noChop.fits"
        files = glob.glob(nom)

        if not files:
            raise FileNotFoundError(f"Aucun fichier trouvé : {nom}")

        oifits_path = files[0]

        with fits.open(oifits_path) as hdu:

            vis2 = hdu["OI_VIS2"].data["VIS2DATA"]
            vis2_err = hdu["OI_VIS2"].data["VIS2ERR"]

            ucoord = hdu["OI_VIS2"].data["UCOORD"]
            vcoord = hdu["OI_VIS2"].data["VCOORD"]

            nrow = ucoord.shape[0]

            lambda_read = hdu["OI_WAVELENGTH"].data["EFF_WAVE"] * 1e6

            hdr = hdu[0].header
            starname.append(hdr.get("HIERARCH ESO OBS TARG NAME", "UNKNOWN"))

            if diam1 is None:
                diam1 = hdr.get("HIERARCH ESO PRO JSDC DIAMETER", np.nan)
                emas = hdr.get("HIERARCH ESO PRO JSDC DIAMETER ERROR", np.nan)

        basename = np.array(extract_baselines(oifits_path))[:N_BASELINES]
        base[idx, :, :] = basename

        wave_mask = (np.abs(lambda_read - LAMBDA_CENTER) < LAMBDA_HALF_WIDTH)
        lamb_sel_um = lambda_read[wave_mask]
        nwave = lamb_sel_um.size

        if nwave == 0:
            raise ValueError("Fenêtre spectrale vide.")

        lamb_sel = lamb_sel_um * 1e-6

        B = np.sqrt(ucoord**2 + vcoord**2)
        spatial_freq = B[:, None] / lamb_sel[None, :]

        if vis2_cube is None:
            vis2_cube = np.zeros((NB_BCD, nrow, nwave))
            errvis2_cube = np.zeros((NB_BCD, nrow, nwave))
            freq_cube = np.zeros((NB_BCD, nrow, nwave))
            lambda_cube_ref = lamb_sel_um.copy()

        vis2_cube[idx] = vis2[:, wave_mask]
        errvis2_cube[idx] = vis2_err[:, wave_mask]
        freq_cube[idx] = spatial_freq
        if idx == 0:
            vis2_initial_cube = np.zeros((NB_BCD, nrow, nwave))
            errvis2_initial_cube = np.zeros((NB_BCD, nrow, nwave))

        vis2_initial_cube[idx] = vis2[:, wave_mask]
        errvis2_initial_cube[idx] = vis2_err[:, wave_mask]

    # baseline courte
    freq_mean_all = np.nanmean(freq_cube, axis=2)
    bcd_short_idx = np.nanargmin(freq_mean_all, axis=1)

    return {
        "vis2_cube": vis2_cube,
        "errvis2_cube": errvis2_cube,
        "vis2_initial_cube": vis2_initial_cube,
        "errvis2_initial_cube": errvis2_initial_cube,
        "freq_cube": freq_cube,
        "lambda_ref": lambda_cube_ref,
        "diam1": diam1,
        "emas": emas,
        "bcd_short_idx": bcd_short_idx,
        "base": base,
        "bcd_states": bcd_states,
    }

def normalize_visibilities(data):

    vis2_cube = data["vis2_cube"]
    errvis2_cube = data["errvis2_cube"]
    bcd_short_idx = data["bcd_short_idx"]

    NB_BCD, nrow, nwave = vis2_cube.shape

    # =========================
    # Normalisation baseline courte
    # =========================

    vis2_norm_cube = np.zeros_like(vis2_cube)
    errvis2_norm_cube = np.zeros_like(errvis2_cube)

    for ii in range(NB_BCD):
        j = int(bcd_short_idx[ii])

        for i in range(nrow):
            vis2_norm_cube[ii, i] = vis2_cube[ii, i] / vis2_cube[ii, j]

            errvis2_norm_cube[ii, i] = np.abs(vis2_norm_cube[ii, i]) * np.sqrt(
                (errvis2_cube[ii, i] / vis2_cube[ii, i])**2 +
                (errvis2_cube[ii, j] / vis2_cube[ii, j])**2
            )

    vis2_cube = vis2_norm_cube.copy()
    errvis2_cube = errvis2_norm_cube.copy()

    # =========================
    # Normalisation spectrale
    # =========================

    i_ref = nwave // 2

    vis_ref_cube = np.zeros_like(vis2_cube)
    errvis_ref_cube = np.zeros_like(errvis2_cube)

    for ii in range(NB_BCD):
        for i in range(nrow):
            vis_ref_cube[ii, i] = vis2_cube[ii, i] / vis2_cube[ii, i, i_ref]
            errvis_ref_cube[ii, i] = errvis2_cube[ii, i] / vis2_cube[ii, i, i_ref]

    # on renvoie tout ce qui sert après
    return {
        "vis_ref_cube": vis_ref_cube,
        "errvis_ref_cube": errvis_ref_cube,
        "i_ref": i_ref,
    }

def prepare_minimization(data, norm):

    freq_cube = data["freq_cube"]
    vis2_initial_cube = data["vis2_initial_cube"]
    errvis2_initial_cube = data["errvis2_initial_cube"]
    lambda_cube_ref = data["lambda_ref"]
    diam1 = data["diam1"]
    emas = data["emas"]
    bcd_short_idx = data["bcd_short_idx"]
    vis_ref_cube = norm["vis_ref_cube"]
    errvis_ref_cube = norm["errvis_ref_cube"]
    i_ref = norm["i_ref"]

    NB_BCD, nrow, nwave = freq_cube.shape

    # =========================
    # Baseline courte
    # =========================
    bs = np.broadcast_to(bcd_short_idx[:, None, None], (NB_BCD, nrow, nwave)).copy()

    # =========================
    # Delta lambda
    # =========================
    delta_lambda_um = (lambda_cube_ref - lambda_cube_ref[i_ref])
    delta_lambda = np.broadcast_to(
        delta_lambda_um[None, None, :],
        (NB_BCD, nrow, nwave)
    )

    # =========================
    # Fréquences spatiales
    # =========================
    spatial_frq = freq_cube

    spatial_frq_bc = np.zeros_like(freq_cube)
    for ii in range(NB_BCD):
        j = int(bcd_short_idx[ii])
        spatial_frq_bc[ii] = freq_cube[ii, j][None, :]

    spatial_frq_ref = freq_cube[:, :, i_ref][:, :, None]

    spatial_frq_ref_bc = np.zeros_like(freq_cube)
    for ii in range(NB_BCD):
        j = int(bcd_short_idx[ii])
        spatial_frq_ref_bc[ii] = freq_cube[ii, j, i_ref]

    # =========================
    # Poids
    # =========================
    weight1 = np.ones((NB_BCD, nrow, nwave))

    # =========================
    # Paramètres initiaux
    # =========================
    b0, c0 = 0.0, 0.0

    bb_min, bb_max = -2.0, 2.0
    cc_min, cc_max = -2.0, 2.0

    diam_min = diam1 - 2.0 * emas
    diam_max = diam1 + 2.0 * emas

    # =========================
    # Tableaux résultats
    # =========================
    model_b = np.zeros((NB_BCD, nrow, nwave))
    model_c = np.zeros((NB_BCD, nrow, nwave))
    diam_init = np.zeros((NB_BCD, nrow, nwave))
    chi2_min_base = np.zeros((NB_BCD, nrow))

    return {
        "spatial_frq": spatial_frq,
        "spatial_frq_bc": spatial_frq_bc,
        "spatial_frq_ref": spatial_frq_ref,
        "spatial_frq_ref_bc": spatial_frq_ref_bc,
        "vis_ref_cube": vis_ref_cube,
        "errvis_ref_cube": errvis_ref_cube,
        "delta_lambda": delta_lambda,
        "weight1": weight1,
        "b0": b0,
        "c0": c0,
        "bb_min": bb_min,
        "bb_max": bb_max,
        "cc_min": cc_min,
        "cc_max": cc_max,
        "diam_min": diam_min,
        "diam_max": diam_max,
        "model_b": model_b,
        "model_c": model_c,
        "diam_init": diam_init,
        "chi2_min_base": chi2_min_base,
        "bs": bs,
        "i_ref": i_ref,
        "vis2_initial_cube": vis2_initial_cube,
        "errvis2_initial_cube": errvis2_initial_cube,
    }

def first_fit(prep, data):

    spatial_frq = prep["spatial_frq"]
    spatial_frq_bc = prep["spatial_frq_bc"]
    spatial_frq_ref = prep["spatial_frq_ref"]
    spatial_frq_ref_bc = prep["spatial_frq_ref_bc"]

    vis_ref_cube = prep["vis_ref_cube"]
    errvis_ref_cube = prep["errvis_ref_cube"]

    delta_lambda = prep["delta_lambda"]
    weight1 = prep["weight1"]

    b0 = prep["b0"]
    c0 = prep["c0"]

    bb_min = prep["bb_min"]
    bb_max = prep["bb_max"]
    cc_min = prep["cc_min"]
    cc_max = prep["cc_max"]

    bs = prep["bs"]
    i_ref = prep["i_ref"]

    diam1 = data["diam1"]

    NB_BCD, nrow, nwave = spatial_frq.shape

    # =============================================================================
    # 1. PREMIER TRI DES DONNÉES SUR LES RÉSIDUS
    # =============================================================================

    result = tri_res(kMAD,spatial_frq,spatial_frq_bc,spatial_frq_ref,spatial_frq_ref_bc,vis_ref_cube,errvis_ref_cube,delta_lambda,weight1,b0,c0,diam1,i_ref,bs)

    weight1 = result["weight"]

    points_gardes_1 = np.sum(weight1 > 0)
    points_rejetes_1 = np.sum(weight1 == 0)
    points_total = weight1.size
    points_attendus = NB_BCD * (nrow - 1) * (nwave - 1)

    chi2_step1 = objective_function_transfer_first_minimization((b0, c0),spatial_frq,spatial_frq_bc,spatial_frq_ref,spatial_frq_ref_bc,vis_ref_cube,errvis_ref_cube,delta_lambda, weight1,diam1)

    # =============================================================================
    # 2. PREMIÈRE MINIMISATION DES PARAMÈTRES DE LA FONCTION DE TRANSFERT
    # =============================================================================

    opt = {"rhobeg": 0.01, "maxiter": 5000, "disp": False, "catol": 1e-6}

    minsuccess_step2 = np.zeros((NB_BCD, nrow), dtype=bool)

    model_b = np.zeros((NB_BCD, nrow, nwave), dtype=float)
    model_c = np.zeros((NB_BCD, nrow, nwave), dtype=float)
    chi2_min_base = np.zeros((NB_BCD, nrow), dtype=float)

    for ii in range(NB_BCD):
        for i in range(nrow):

            x0 = np.array([b0, c0], dtype=float)

            constraints = [
                {"type": "ineq", "fun": lambda x, low=bb_min: x[0] - low},
                {"type": "ineq", "fun": lambda x, up=bb_max: up - x[0]},
                {"type": "ineq", "fun": lambda x, low=cc_min: x[1] - low},
                {"type": "ineq", "fun": lambda x, up=cc_max: up - x[1]},
            ]

            res1 = minimize(objective_function_transfer_first_minimization,x0,method="COBYLA",args=(spatial_frq[ii, i, :],spatial_frq_bc[ii, i, :],spatial_frq_ref[ii, i, :],spatial_frq_ref_bc[ii, i, :],vis_ref_cube[ii, i, :],errvis_ref_cube[ii, i, :],delta_lambda[ii, i, :],weight1[ii, i, :],diam1),constraints=constraints,options=opt)
            

            model_b[ii, i, :] = res1.x[0]
            model_c[ii, i, :] = res1.x[1]

            chi2_min_base[ii, i] = objective_function_transfer_first_minimization(res1.x,spatial_frq[ii, i, :],spatial_frq_bc[ii, i, :],spatial_frq_ref[ii, i, :],spatial_frq_ref_bc[ii, i, :],vis_ref_cube[ii, i, :],errvis_ref_cube[ii, i, :],delta_lambda[ii, i, :],weight1[ii, i, :],diam1)

            minsuccess_step2[ii, i] = res1.success

    return {
        "weight1": weight1,
        "model_b": model_b,
        "model_c": model_c,
        "chi2_min_base": chi2_min_base,
        "minsuccess_step2": minsuccess_step2,
        "chi2_step1": chi2_step1,
    }


def final_fit(prep, fit1, data):

    spatial_frq = prep["spatial_frq"]
    spatial_frq_bc = prep["spatial_frq_bc"]
    spatial_frq_ref = prep["spatial_frq_ref"]
    spatial_frq_ref_bc = prep["spatial_frq_ref_bc"]

    vis_ref_cube = prep["vis_ref_cube"]
    errvis_ref_cube = prep["errvis_ref_cube"]
    vis2_initial_cube = prep["vis2_initial_cube"]
    errvis2_initial_cube = prep["errvis2_initial_cube"]
    delta_lambda = prep["delta_lambda"]
    bs = prep["bs"]
    i_ref = prep["i_ref"]

    bb_min = prep["bb_min"]
    bb_max = prep["bb_max"]
    cc_min = prep["cc_min"]
    cc_max = prep["cc_max"]
    diam_min = prep["diam_min"]
    diam_max = prep["diam_max"]

    model_b = fit1["model_b"]
    model_c = fit1["model_c"]
    weight1 = fit1["weight1"]
    minsuccess_step2 = fit1["minsuccess_step2"]

    diam1 = data["diam1"]

    NB_BCD, nrow, nwave = spatial_frq.shape

    # =============================================================================
    # 3. SECOND TRI
    # =============================================================================

    weight2 = np.ones((NB_BCD, nrow, nwave), dtype=float)

    result2 = tri_res(
        kMAD,
        spatial_frq,
        spatial_frq_bc,
        spatial_frq_ref,
        spatial_frq_ref_bc,
        vis_ref_cube,
        errvis_ref_cube,
        delta_lambda,
        weight2,
        model_b,
        model_c,
        diam1,
        i_ref,
        bs,
    )

    weight2 = result2["weight"]

    valid_bases_step3 = np.ones((NB_BCD, nrow), dtype=bool)

    for ii in range(NB_BCD):
        j_ref = int(bs[ii, 0, 0])

        for i in range(nrow):

            if i == j_ref:
                weight2[ii, i, :] = 0.0
                valid_bases_step3[ii, i] = False
                continue

            b_value = model_b[ii, i, 0]
            c_value = model_c[ii, i, 0]

            b_at_limit = (
                abs(b_value - bb_min) / abs(b_value + bb_min) < 0.001
                or abs(b_value - bb_max) / abs(b_value + bb_max) < 0.001
            )

            c_at_limit = (
                abs(c_value - cc_min) / abs(c_value + cc_min) < 0.001
                or abs(c_value - cc_max) / abs(c_value + cc_max) < 0.001
            )

            reject_base = (
                b_at_limit
                or c_at_limit
                or (not minsuccess_step2[ii, i])
            )

            if reject_base:
                weight2[ii, i, :] = 0.0
                valid_bases_step3[ii, i] = False

 # =============================================================================
    # 4. FIT DIAMÈTRE
    # =============================================================================

    opt = {"rhobeg": 0.01, "maxiter": 5000, "disp": False, "catol": 1e-6}

    model_b2 = np.zeros((NB_BCD, nrow, nwave), dtype=float)
    model_c2 = np.zeros((NB_BCD, nrow, nwave), dtype=float)
    diam_init = np.zeros((NB_BCD, nrow, nwave), dtype=float)

    minsuccess_step4 = np.zeros((NB_BCD, nrow), dtype=bool)

    for ii in range(NB_BCD):
        for i in range(nrow):

            x0 = np.array([0.0, 0.0, diam1], dtype=float)

            constraints = [
                {"type": "ineq", "fun": lambda x: x[0] - bb_min},
                {"type": "ineq", "fun": lambda x: bb_max - x[0]},
                {"type": "ineq", "fun": lambda x: x[1] - cc_min},
                {"type": "ineq", "fun": lambda x: cc_max - x[1]},
                {"type": "ineq", "fun": lambda x: x[2] - diam_min},
                {"type": "ineq", "fun": lambda x: diam_max - x[2]},
            ]

            res2 = minimize(
                objective_function_transfer_and_diameter_second_minimization,
                x0,
                method="COBYLA",
                args=(
                    spatial_frq[ii, i, :],
                    spatial_frq_bc[ii, i, :],
                    spatial_frq_ref[ii, i, :],
                    spatial_frq_ref_bc[ii, i, :],
                    vis_ref_cube[ii, i, :],
                    errvis_ref_cube[ii, i, :],
                    delta_lambda[ii, i, :],
                    weight2[ii, i, :],
                ),
                constraints=constraints,
                options=opt,
            )

            model_b2[ii, i, :] = res2.x[0]
            model_c2[ii, i, :] = res2.x[1]
            diam_init[ii, i, :] = res2.x[2]

            minsuccess_step4[ii, i] = res2.success
            
    # =============================================================================
    # FILTRAGE FINAL 
    # =============================================================================

    valid_bases_step4 = np.ones((NB_BCD, nrow), dtype=bool)

    for ii in range(NB_BCD):
        j_ref = int(bs[ii, 0, 0])

        for i in range(nrow):

            if i == j_ref:
                valid_bases_step4[ii, i] = False
                continue

            diam_value = diam_init[ii, i, 0]
            b_value = model_b2[ii, i, 0]
            c_value = model_c2[ii, i, 0]

            diam_at_limit = (
                abs(diam_value - diam_min) / abs(diam_value + diam_min) < 0.001
                or abs(diam_value - diam_max) / abs(diam_value + diam_max) < 0.001
            )

            b_at_limit = (
                abs(b_value - bb_min) / abs(b_value + bb_min) < 0.001
                or abs(b_value - bb_max) / abs(b_value + bb_max) < 0.001
            )

            c_at_limit = (
                abs(c_value - cc_min) / abs(c_value + cc_min) < 0.001
                or abs(c_value - cc_max) / abs(c_value + cc_max) < 0.001
            )

            reject_base = (
                b_at_limit
                or c_at_limit
                or (not minsuccess_step4[ii, i])
            )

            if reject_base:
                weight2[ii, i, :] = 0.0
                valid_bases_step4[ii, i] = False


    # =============================================================================
    # 5. DIAM GLOBAL
    # =============================================================================

    x0 = np.array([diam1], dtype=float)

    constraints = [
        {"type": "ineq", "fun": lambda x: x[0] - diam_min},
        {"type": "ineq", "fun": lambda x: diam_max - x[0]},
    ]

    res3 = minimize(
        objective_function_transfer_bc_fixed_third_minimization,
        x0,
        method="COBYLA",
        args=(
            spatial_frq,
            spatial_frq_bc,
            spatial_frq_ref,
            spatial_frq_ref_bc,
            vis_ref_cube,
            errvis_ref_cube,
            delta_lambda,
            weight2,
            model_b2,
            model_c2,
        ),
        constraints=constraints,
        options=opt,
    )

    diam_global = res3.x[0]

    chi2_global = objective_function_transfer_bc_fixed_third_minimization(
        np.array([diam_global]),
        spatial_frq,
        spatial_frq_bc,
        spatial_frq_ref,
        spatial_frq_ref_bc,
        vis_ref_cube,
        errvis_ref_cube,
        delta_lambda,
        weight2,
        model_b2,
        model_c2,
    )

    return {
    "diam_global": diam_global,
    "chi2_global": chi2_global,
    "diam_candidates": diam_init,
    "valid_bases": valid_bases_step4,

    "model_b": model_b,
    "model_c": model_c,

    "weight_final": weight2,

    "vis_ref_cube": vis_ref_cube,
    "vis2_initial_cube": vis2_initial_cube,
    "errvis2_initial_cube": errvis2_initial_cube,
    "freq_cube": spatial_frq,
    "lambda_ref": data["lambda_ref"],

    "success": res3.success,
    }

def run_minimization(data):

    norm = normalize_visibilities(data)
    prep = prepare_minimization(data, norm)
    fit1 = first_fit(prep, data)
    final = final_fit(prep, fit1, data)

    return final