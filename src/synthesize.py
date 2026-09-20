
import numpy as np
import healpy as hp
import pysm3 as sm

from src.anisotropy import calculate_wlm

def get_needlets(l_max, fwhms_arcmin=[300., 120., 60., 30., 15., 7.5, 5.]):
    """get_needlets(l_max, fwhm_arcmin)
    
    Returns Gaussian-based band-pass filters based on Gaussian FWHMs.

    Args:
        l_max (int): Maximum ell
        fwhms_arcmin (tuple): Gaussian FWHM in arcmin
    
    Returns:
        np.ndarray: Array of band-pass filters at scales
    """
    l = np.arange(l_max+1)
    num_bands = len(fwhms_arcmin)+1
    fwhms_arcmin = np.array(fwhms_arcmin)
    fwhm_rad = fwhms_arcmin * (np.pi / (180.*60.))
    beams = np.zeros((len(fwhms_arcmin), len(l)))
    for j in range(len(fwhms_arcmin)):
        beams[j] = hp.gauss_beam(fwhm_rad[j], l_max)
    h_l = np.zeros((num_bands, len(l)))
    h_l[0] = beams[0]
    for j in range(1, len(fwhms_arcmin)):
        diff = beams[j]**2 - beams[j-1]**2
        diff[diff < 0.] = 0.
        h_l[j] = np.sqrt(diff)
    diff_last = 1.0 - beams[-1]**2
    diff_last[diff_last < 0.] = 0.
    h_l[-1] = np.sqrt(diff_last)
    return h_l

def ContinuousCascadeSphere(
        l_max, NSIDE,
        omega_cl, field_cl,
        do_lagrangian_map,
        needlet_fwhms=[300., 120., 60., 30., 15., 7.5, 5.],
        sigma_param=0.2, c_param=5.0, params=None):
    """
    Builds the multifractal scalar potential scale-by-scale.
    If do_lagrangian_map is True, tracks and returns advected grid coordinates.
    Otherwise, returns the raw unmapped scalar potential.

    Args:
        l_max (int): Maximum ell
        NSIDE (int): HEALPiX N_side
        omega_cl (np.ndarray): C_ell for the multiplicative noise
        field_cl (np.ndarray): C_ell for the base field
        do_lagrangian_map (bool): If true, return advected grid coordinates
        needlet_fwhms (tuple): Gaussian FWHM in arcmin for scale selection
        sigma_param (float): (lambda^2), the multifractal parameter
        c_param (float): Advection strength parameter
        params (tuple): (s_l, s_m, theta_deg) anisotropy transformation parameters

    Returns:
        (np.ndarray, np.ndarray): Advected grid coordinates d_theta, d_phi
        np.ndarray: Scalar field
    """
    # Map(s)
    npix = hp.nside2npix(NSIDE)
    omega = np.zeros(npix)  # Intensity process (intermittency)
    A_map = np.zeros(npix)  # Accumulated scalar potential

    # Needlet/scale-dependence
    h_l = get_needlets(l_max, needlet_fwhms)
    num_bands = len(needlet_fwhms)
    
    # Grid coordinates and displacements
    theta, phi = hp.pix2ang(NSIDE, np.arange(npix))
    theta_disp = np.zeros(npix)
    phi_disp = np.zeros(npix)

    # Field Cls
    l = np.arange(l_max+1)
    pot_cl = field_cl.copy()
    pot_cl[np.isnan(pot_cl)] = 0.
    pot_cl[:2] = 0.
    om_cl = omega_cl.copy()
    om_cl[np.isnan(om_cl)] = 0.
    om_cl[:2] = 0.
    
    for j in range(num_bands):
        weight_sum = np.sum(h_l[j])
        l_mid = np.sum(l * h_l[j]) / weight_sum
        
        # A. Sample next level of the intensity process (Omega)
        # Generate white noise, filter by needlet, and accumulate
        alm_Omega = hp.synalm(om_cl, l_max)
        alm_Omega = hp.almxfl(alm_Omega, h_l[j])
        Omega = hp.alm2map(alm_Omega, nside=NSIDE)
        Omega = (Omega - np.nanmean(Omega)) / np.nanstd(Omega)
        omega += np.sqrt(sigma_param)*Omega - (sigma_param / 2.)
        
        # B. Sample next level of the potential
        alm_a_noise = hp.synalm(pot_cl, lmax=l_max)
        if params is not None:
            wlm = calculate_wlm(l_max, [pot_cl,], s_l=params[0], s_m=params[1], theta_deg=params[2])[0]
            alm_a_noise = alm_a_noise*wlm
        alm_a_band = hp.almxfl(alm_a_noise, h_l[j])
        a_band_map = hp.alm2map(alm_a_band, NSIDE)
        
        # C. Current step of scale integral (modulated by e^omega)
        # We apply a base l_mid scaling here; spectral slope is strictly enforced later
        A_map += np.exp(omega) * a_band_map
        
        # D. Lagrangian Advection Step
        if do_lagrangian_map:
            # Take curl of the ACCUMULATED potential (as per pseudocode v <- curl(a))
            A_alm_current = hp.map2alm(A_map, lmax=l_max)
            _, v_theta, v_phi_sin = hp.alm2map_der1(A_alm_current, NSIDE)
            
            u_theta = v_phi_sin
            u_phi = -v_theta
            
            # Advect coordinates proportional to velocity / max(||v||)
            v_mag = np.sqrt(u_theta**2 + u_phi**2)
            max_v = np.max(v_mag)
            
            if max_v > 0:
                tau = c_param / max_v / l_mid
                theta_disp += tau * u_theta
                phi_disp += tau * u_phi / np.maximum(np.sin(theta + theta_disp), 1e-15)

    if do_lagrangian_map:
        # Return displacements
        # Resolve Spherical Boundaries for advected coordinates
        theta_new = theta + theta_disp
        phi_new = phi + phi_disp
        
        np_cross = theta_new < 0
        theta_new[np_cross] = -theta_new[np_cross]
        phi_new[np_cross] += np.pi
        
        sp_cross = theta_new > np.pi
        theta_new[sp_cross] = 2 * np.pi - theta_new[sp_cross]
        phi_new[sp_cross] += np.pi
        
        phi_new = phi_new % (2 * np.pi)
        return theta_new, phi_new
    else:
        # Return scalar field
        return A_map

def whiten_alm(alm):
    """whiten_alm(alm)

    Normalizes the alm to white noise

    Args:
        alm (np.ndarray): Spherical harmonic coefficients (a_lm)

    Returns:
        np.ndarray: Normalized spherical harmonic coefficients
    """
    cl_emp = hp.alm2cl(alm)
    cl_emp[cl_emp == 0] = 1e-15 # Avoid divide by zero
    return hp.almxfl(alm, 1.0 / np.sqrt(cl_emp))

def get_alm(
        l_max, nside,
        base, base_cl, base_om_cl, base_sigma,
        prop, prop_cl, prop_om_cl, prop_sigma, prop_c):
    """get_alms()
    
    Generates a_lm with the given parameters. Then whitens the a_lm.

    base/prop are given as 'isotropic', 'meridional', or 'zonal'.

    Args:
        l_max (int): Maximum ell
        nside (int): HEALPix N_side
        base (str): What anisotropies (if any) to generate for the base field
        base_cl (np.ndarray): C_ell of the base field
        base_om_cl (np.ndarray): C_ell of the base field's multiplicative noise
        base_sigma (float): Base field multifractal parameter

        prop (str): What anisotropies (if any) to generate for the displacement field
        prop_cl (np.ndarray): C_ell of the displacement field
        prop_om_cl (np.ndarray): C_ell of the displacement field multiplicative noise
        prop_sigma (float): Displacement field multifractal parameter
        prop_c (float): Displacement field advection strength
    
    Returns:
        np.ndarray: single whitened realizations of the field with given parameters
    """
    ## We need the same propagation for each field
    if prop == 'isotropic':
        theta, phi = ContinuousCascadeSphere(l_max, nside, prop_om_cl, prop_cl, True, sigma_param=prop_sigma, c_param=prop_c)
    elif prop == 'meridional':
        theta, phi = ContinuousCascadeSphere(l_max, nside, prop_om_cl, prop_cl, True, sigma_param=prop_sigma, c_param=prop_c, params=(20.0, 1.0, 45.0))
    elif prop == 'zonal':
        theta, phi = ContinuousCascadeSphere(l_max, nside, prop_om_cl, prop_cl, True, sigma_param=prop_sigma, c_param=prop_c, params=(1.0, 10.0, 0.0))
    else:
        raise NotImplementedError('Have not implemented other anisotropies')

    ## We will have the same base (but different realizations) for each field
    if base == 'isotropic':
        map = ContinuousCascadeSphere(l_max, nside, base_om_cl, base_cl, False, sigma_param=base_sigma)
    elif base == 'meridional':
        map = ContinuousCascadeSphere(l_max, nside, base_om_cl, base_cl, False, sigma_param=base_sigma, params=(20.0, 1.0, 45.0))
    elif base == 'zonal':
        map = ContinuousCascadeSphere(l_max, nside, base_om_cl, base_cl, False, sigma_param=base_sigma, params=(1.0, 10.0, 0.0))
    else:
        raise NotImplementedError('Have not implemented other anisotropies')
    # Displace the base field by the displacements
    w = hp.get_interp_val(map, theta, phi)
    alm = hp.map2alm(w, lmax=l_max)
    return whiten_alm(alm)

def make_iqu(
        l_max, nside,
        t_delta_params,
        e_delta_params,
        b_delta_params,
        cl_tt, cl_ee, cl_bb, cl_te,
        seeds=(0,1,2),
):
    """make_iqu()

    Make small-scale polarization tensor maps i_delta, q_delta, u_delta.
    
    """
    alms = []
    for i in range(3):
        ## Assume parameters are equal for displacement map and base map
        param_dict = ([t_delta_params, e_delta_params, b_delta_params])[i]
        target_cl = param_dict['cl_target']
        noise_cl = param_dict['cl_noise']
        c_param = param_dict['c']
        lambda_param = param_dict['lambda']
        ani_str = param_dict['anisotropy']

        np.random.seed(seeds[i])
        alm = get_alm(l_max, nside, ani_str, target_cl, noise_cl, lambda_param**2, ani_str, target_cl, noise_cl, lambda_param**2, c_param)
        alms.append(alm)

    ## Use analytical solution for zero tb, eb correlations
    ## Easy enough to replace with true Cholesky decomposition
    # # t-mode
    weight_T1 = np.sqrt(cl_tt)
    alm_T = hp.almxfl(alms[0], weight_T1)
    # # e-mode
    weight_E1 = cl_te / np.sqrt(cl_tt)
    weight_E2 = np.sqrt(cl_ee - (cl_te**2 / cl_tt))
    alm_E = hp.almxfl(alms[0], weight_E1) + hp.almxfl(alms[1], weight_E2)
    # # b-mode
    weight_B3 = np.sqrt(cl_bb)
    alm_B = hp.almxfl(alms[2], weight_B3)

    I_map, Q_map, U_map = hp.alm2map((alm_T, alm_E, alm_B), nside, pol=True)
    return I_map, Q_map, U_map

def make_IQU_ref(i_delta, q_delta, u_delta, nside=512):
    """Transform given i,q,u polarization tensor quantities into full-sky all-scale I,Q,U maps using pysm-s6"""
    sky6 = sm.Sky(nside=nside, preset_strings=['s6'])
    I_ref, Q_ref, U_ref, _ = sky6.components[0].modulate_small_scales(np.array([i_delta, q_delta, u_delta]), 3*nside-1, (4,5,6))
    return I_ref, Q_ref, U_ref

def get_apodized_mask(nside=512):
    """Returns apodized mask for Galactic plane synthesis"""
    masks = hp.read_map('HFI_Mask_GalPlane-apo2_2048_R2.00.fits', nest=True, hdu=1, field=(0,1,2,3,4,5,6,7))
    masks = hp.ud_grade(masks, nside, order_in='NEST', order_out='RING')
    mask_apodized = masks[3]
    return mask_apodized

def blur_maps(map1, map2, mask_apodized=None, nside=512):
    """Mix 2 maps based on apodized mask"""
    if mask_apodized is None:
        mask_apodized = get_apodized_mask(nside)
    map = (1.-mask_apodized)*map1 + mask_apodized*map2
    return map

def blur_alms(alm1, alm2, mask_apodized=None, l_max=1535, nside=512):
    if mask_apodized is None:
        mask_apodized = get_apodized_mask(nside)
    map1 = hp.alm2map(alm1, nside)
    map2 = hp.alm2map(alm2, nside)
    map = (1.-mask_apodized)*map1 + mask_apodized*map2
    alm = hp.map2alm(map, l_max)
    return whiten_alm(alm)

