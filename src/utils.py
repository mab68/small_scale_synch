
import numpy as np
import scipy.fft as fft
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy import wcs

def modify_rc():
    mpl.rcParams['figure.dpi'] = 250
    mpl.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r'''\usepackage{bm}
\usepackage{xcolor}''')
    #mpl.rcParams['text.latex.preamble']=[r"\usepackage{bm}", r"\usepackage{xcolor}"]
    mpl.rc('font', family='serif', serif='Computer Modern', size=8)

# NOTE: COMMENT THIS OUT IF YOU DON'T HAVE LATEX CONFIGURED FOR MATPLOTLIB
modify_rc()
modify_rc()

def pretty_axes(axx):
    axx.yaxis.set_ticks_position('both')
    axx.xaxis.set_ticks_position('both')
    axx.tick_params(axis='y', direction='in')
    axx.tick_params(axis='y', direction='in', which='minor')
    axx.tick_params(axis='x', direction='in')
    axx.tick_params(axis='x', direction='in', which='minor')
    axx.grid(linestyle=':', alpha=0.3, linewidth=0.5, color='gray')

def get_minmax(ar, p=0, sym=True):
    if p != 0:
        cmin, cmax = np.nanpercentile(ar, p), np.nanpercentile(ar, 100-p)
    else:
        cmin, cmax = np.nanmin(ar), np.nanmax(ar)
    if sym:
        vmax = np.nanmax([np.abs(cmin), np.abs(cmax)])
        vmin = -vmax
    else:
        vmin, vmax = cmin, cmax
    return vmin, vmax

def patch_creation(n_pix, delta_pix, pos_long, pos_lat):
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [n_pix / 2, n_pix / 2]
    w.wcs.cdelt = np.array([-delta_pix, delta_pix])
    w.wcs.crval = [pos_long, pos_lat]
    w.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]
    patch = np.zeros((n_pix, n_pix))
    return (w, patch)

def fill_patch(w, patch, map_fill):
    patch_index = np.indices(np.shape(patch))
    lon, lat = w.wcs_pix2world(patch_index[1], patch_index[0], 0)
    get_pix_sky = hp.ang2pix(hp.get_nside(map_fill), lon, lat, lonlat=True)
    all_pix_values = map_fill[get_pix_sky]
    filled_patch = np.reshape(all_pix_values, np.shape(patch))
    return filled_patch

def w_object(n_pix, delta_pix, pos_long, pos_lat):
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [n_pix / 2, n_pix / 2]
    w.wcs.cdelt = np.array([-delta_pix, delta_pix])
    w.wcs.crval = [pos_long, pos_lat]
    w.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]
    return w

def get_patch_data(field, l_center, b_center, size_deg=32., res_arcmin=4.94):
    """
    Extracts a 2D patch from a HEALPix map.
    Assumes the input field is already in Galactic coordinates (G).
    """
    half_size = size_deg / 2.0
    
    # 1. Calculate boundaries based on center and size
    l_min, l_max = l_center - half_size, l_center + half_size
    b_min, b_max = b_center - half_size, b_center + half_size
    
    # Clamp latitudes to physical limits
    b_min, b_max = max(b_min, -90.0), min(b_max, 90.0)

    if l_center + half_size > 360:
        l_min = l_min - 360 if l_min > 180 else l_min
        l_max = l_max - 360 if l_max > 180 else l_max
        l_min, l_max = min(l_min, l_max), max(l_min, l_max)

    xsize = int(np.round((size_deg * 60.0) / res_arcmin))

    grid_data = hp.gnomview(
        field,
        rot=[l_center, b_center],
        reso=res_arcmin,
        xsize=xsize,
        ysize=xsize,
        return_projected_map=True
    )
    plt.close()

    num_b, num_l = grid_data.shape

    l_edges = np.linspace(l_min, l_max, num_l + 1)
    b_edges = np.linspace(b_max, b_min, num_b + 1)
    l_edges = np.where(l_edges < 0, l_edges + 360, l_edges)

    return l_edges, b_edges, grid_data
