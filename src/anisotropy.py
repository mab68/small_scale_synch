
import numpy as np
import healpy as hp

def get_anisotropic_l_eff(l_vec, m_vec, s_l=1., s_m=1., theta_deg=0.):
    """get_anisotropic_l_eff(l_vec, m_vec, s_l, s_m, theta_deg)\n

    Returns the transformed ells (l_eff)

    Args:
        l_vec (np.ndarray): a_lm shaped grid corresponding to ells
        m_vec (np.ndarray): a_lm shaped grid corresponding to m's
        s_l (float): ell-mode stretching parameter
        s_m (float): m-mode stretching parameter
        theta_deg (float): Rotation matrix parameter (in degress)    

    Returns:
        np.ndarray: a_lm shaped grid mapping the effective ells
    """
    ## Define the transformation matrix
    theta = np.radians(theta_deg)
    # rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    # scaling matrix
    S = np.array([
        [s_l, 0.],
        [0., s_m],
    ])
    # transformation matrix
    G = S @ R
    ## Stack and apply transformation
    lm_stack = np.vstack((l_vec, m_vec))
    transformed_lm = G @ lm_stack
    # Compute 2D elliptical distance
    l_prime = transformed_lm[0]
    m_prime = transformed_lm[1]
    # Get effective ells
    l_eff = np.sqrt(l_prime**2 + m_prime**2)
    return l_eff

def calculate_wlm(l_max, target_cls, s_l, s_m, theta_deg=0.):
    """calculate_wlm(l_max, Cl, s_l, s_m, theta_deg)\n
    
    Returns the power-rescaling factor W_lm = W_lm^anisotropic * W_lm^isotropic

    Args:
        l_max (float): Maximum ell
        target_cls (tuple): List of target_Cls
        s_l (float): l-mode stretching parameter
        s_m (float): m-mode stretching parameter
        theta_deg (float): Rotation matrix parameter (in degrees)
    
    Returns:
        np.ndarray: a_lm shaped grid rescaling factor
    """
    l_vec, m_vec = hp.Alm.getlm(l_max)
    l_eff = get_anisotropic_l_eff(l_vec, m_vec, s_l, s_m, theta_deg)
    num_cls = np.shape(target_cls)[0]
    num_ells = np.shape(target_cls)[1]
    ell_integers = np.arange(num_ells)
    w_lm = []
    for i in range(num_cls):
        ## The target Cl
        Cl_base = np.interp(l_vec, ell_integers, target_cls[i], left=0., right=0.)
        ## The effective (anisotropic) Cl
        Cl_eff = np.interp(l_eff, ell_integers, target_cls[i], left=0., right=0.)
        ## Get anisotropic factor -- multiplying alms with this factor makes the field anisotropic
        ## However, this will modify the isotropic Cl
        w_lm2 = np.zeros_like(l_eff)
        np.divide(Cl_eff, Cl_base, out=w_lm2, where=(Cl_base != 0.))

        ## Get the isotropic factor to renormalize the alm to give the correct isotropic Cl
        m_multiplier = np.ones_like(m_vec)
        m_multiplier[m_vec > 0] = 2.
        power_contributions = m_multiplier * Cl_eff #hp.almxfl(m_multiplier, Cl_eff)
        summed_power = np.bincount(l_vec, weights=np.abs(power_contributions))
        ## This is the power of the anisotropic field
        valid_ells = np.arange(len(summed_power))
        Cl_expected = summed_power / (2. * valid_ells + 1.)
        ## Now get the factor to convert to the isotropic Cl
        w_iso2 = np.zeros_like(Cl_expected)
        Cl_base1d = np.interp(valid_ells, ell_integers, target_cls[i], left=0., right=0.)
        np.divide(Cl_base1d, Cl_expected, out=w_iso2, where=(Cl_expected > 0.))

        w_lm.append(hp.almxfl(np.sqrt(w_lm2), np.sqrt(w_iso2)))

    return w_lm
