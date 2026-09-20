
import gc

import numpy as np
import healpy as hp

from scipy.spatial import cKDTree

def sph_structure_function(scalar_field, r_bins, phi_bins, num_anchors=None, chunk_size=5000, powers=[2,]):
    """sph_structure_function(scalar_field, r_bins, phi_bins, num_anchors, chunk_size, powers)

    Estimates the p-th order structure function for data on the sphere.

    Args:
        scalar_field (np.ndarray): HEALPiX map
        r_bins (np.ndarray): Bins for the lag distance
        phi_bins (np.ndarray): Bins for the angular distance
        num_anchors (int): Number of points to sub-sample from the sphere
        chunk_size (int): Size of the chunk to process anchors
        powers (tuple): List of structure function moments to compute
    
    Returns:
        np.ndarray, np.ndarray, dict: distance bins, angle bins, and p-th order SF (indexed by order p)
    """
    # Get scalar field vectors/coordinates
    npix = len(scalar_field)
    nside = hp.npix2nside(npix)
    pix_indices = np.arange(npix)
    x, y, z = hp.pix2vec(nside, pix_indices)
    all_coords = np.column_stack((x,y,z))

    # Sub-sample the pixels (so we don't compute everything, but some sub-set for increased speed)
    if num_anchors is not None and num_anchors < npix:
        anchors = np.random.choice(npix, size=num_anchors, replace=False)
    else:
        anchors = pix_indices
    anchor_coords = all_coords[anchors]

    # Find all neighbours within a maximum radial distance
    min_r_rad = r_bins[0]
    max_r_rad = r_bins[-1]
    min_chord_sq = (2.*np.sin(min_r_rad / 2.))**2
    max_chord = 2.*np.sin(max_r_rad / 2.)

    # Find pairs between anchors and all pixels
    # Returns for each anchor, the indices of its neighbours
    print('Generating KD Tree')
    tree_all = cKDTree(all_coords)

    # Initialize 2D histograms to accumulate results in chunks
    sf_sum = {p: np.zeros((len(r_bins) - 1, len(phi_bins) - 1)) for p in powers}
    counts = np.zeros((len(r_bins) - 1, len(phi_bins) - 1))

    ## Pre-calculate local headings for all anchors
    theta_anchors, phi_anchors = hp.pix2ang(nside, anchors)
    cos_theta_a = np.cos(theta_anchors)
    sin_theta_a = np.sin(theta_anchors)
    cos_phi_a = np.cos(phi_anchors)
    sin_phi_a = np.sin(phi_anchors)

    # Process anchors in chunks to save memory overhead
    num_anchors_total = len(anchors)
    print('Processing %s anchors in chunks of %s' % (num_anchors_total, chunk_size))
    for chunk_start in range(0, num_anchors_total, chunk_size):
        print('Chunk: ', chunk_start)
        chunk_end = min(chunk_start + chunk_size, num_anchors_total)
        chunk_anchors = anchors[chunk_start:chunk_end]
        chunk_coords = anchor_coords[chunk_start:chunk_end]

        chunk_pairs = tree_all.query_ball_point(chunk_coords, r=max_chord)

        lengths = np.array([len(p) for p in chunk_pairs])
        if len(lengths) == 0 or np.sum(lengths) == 0:
            continue

        # Generate anchor and neighbour arrays
        idx1 = np.repeat(chunk_anchors, lengths)
        idx2 = np.concatenate(chunk_pairs).astype(np.int64)

        # Remove self-matches
        valid_mask = idx1 != idx2
        if not np.any(valid_mask):
            continue
        idx1 = idx1[valid_mask]
        idx2 = idx2[valid_mask]

        # Get chunk-relative indices
        chunk_rel_idx = np.repeat(np.arange(chunk_start, chunk_end), lengths)[valid_mask]
        
        # Vectorized geometries
        vec1 = all_coords[idx1]
        vec2 = all_coords[idx2]

        # Calculate deltas
        dx = vec2[:,0] - vec1[:,0]
        dy = vec2[:,1] - vec1[:,1]
        dz = vec2[:,2] - vec1[:,2]

        # Filter out features that are within a certain radius
        if min_r_rad > 0.:
            chord_sq = dx**2 + dy**2 + dz**2
            inside_band = chord_sq >= min_chord_sq

            if not np.any(inside_band):
                continue

            idx1 = idx1[inside_band]
            idx2 = idx2[inside_band]
            vec1 = vec1[inside_band]
            vec2 = vec2[inside_band]
            dx = dx[inside_band]
            dy = dy[inside_band]
            dz = dz[inside_band]
            chunk_rel_idx = chunk_rel_idx[inside_band]

        # Geodesic indices
        cos_r = np.sum(vec1 * vec2, axis=1)
        r_arr = np.arccos(np.clip(cos_r, -1.0, 1.0))

        # Get pre-calculated trig values
        ct = cos_theta_a[chunk_rel_idx]
        st = sin_theta_a[chunk_rel_idx]
        cp = cos_phi_a[chunk_rel_idx]
        sp = sin_phi_a[chunk_rel_idx]

        proj_north = (ct * cp * dx) + (ct * sp * dy) - (st * dz)
        proj_east = (-sp * dx) + (cp * dy)

        # Get direction
        phi_arr = np.arctan2(proj_east, proj_north)
        np.mod(phi_arr, np.pi, out=phi_arr)

        # Count the pairs in each bin
        chunk_counts, _, _ = np.histogram2d(r_arr, phi_arr, bins=[r_bins, phi_bins])
        counts += chunk_counts

        # Compute difference
        base_diff = np.abs(scalar_field[idx1] -  scalar_field[idx2])
        # Compute SF of order p
        for p in powers:
            diff_p = base_diff**p
            chunk_sf_sum, _, _ = np.histogram2d(r_arr, phi_arr, bins=[r_bins, phi_bins], weights=diff_p)

            sf_sum[p] += chunk_sf_sum

        del chunk_pairs, idx1, idx2, vec1, vec2, dx, dy, dz, proj_north, proj_east, r_arr, phi_arr, base_diff
        gc.collect()

    # Compute the final average SF
    for p in powers:
        sf_sum[p] = sf_sum[p]/counts

    return r_bins, phi_bins, sf_sum
