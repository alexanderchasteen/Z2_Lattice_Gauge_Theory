import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from scipy.special import erf
from sklearn.metrics import r2_score


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2*sigma**2))



def hist_plot(data, weights=None, bins=50, threshold=0.7, n_blocks=50, show_halves=False):
    """
    Plot weighted histogram, fit Gaussians to left/right halves, compute R^2,
    and estimate jackknife errors on peak position, FWHM, and Gaussian amplitude.
    """
    counts, bin_edges = np.histogram(data, bins=bins, weights=weights)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = bin_edges[1] - bin_edges[0]

    # --- Gaussian fit helper ---
    def fit_gaussian(mask, side_name, counts_to_fit):
        try:
            popt, _ = curve_fit(
                gaussian,
                bin_centers[mask],
                counts_to_fit[mask],
                p0=[np.max(counts_to_fit[mask]), bin_centers[mask][np.argmax(counts_to_fit[mask])], 0.01]
            )
            A, mu, sigma = popt
            fwhm = 2.355 * sigma
            residuals = counts_to_fit[mask] - gaussian(bin_centers[mask], *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((counts_to_fit[mask] - np.mean(counts_to_fit[mask]))**2)
            r2 = 1 - ss_res / ss_tot
            return popt, A, mu, fwhm, r2
        except RuntimeError:
            print(f"Gaussian fit failed for {side_name} side.")
            return [np.nan]*3, np.nan, np.nan, np.nan, np.nan

    left_mask = bin_centers < threshold
    right_mask = bin_centers > threshold

    # Fit full data Gaussians
    popt_left, A_left, mu_left, fwhm_left, r2_left = fit_gaussian(left_mask, "Left", counts)
    popt_right, A_right, mu_right, fwhm_right, r2_right = fit_gaussian(right_mask, "Right", counts)

    print(f"Left Gaussian: peak = {A_left:.5f}, mu = {mu_left:.5f}, FWHM = {fwhm_left:.5f}, R^2 = {r2_left:.5f}")
    print(f"Right Gaussian: peak = {A_right:.5f}, mu = {mu_right:.5f}, FWHM = {fwhm_right:.5f}, R^2 = {r2_right:.5f}")

    # --- Jackknife for mu, FWHM, peak ---
    N = len(data)
    block_size = N // n_blocks
    mu_left_jk, fwhm_left_jk, A_left_jk = [], [], []
    mu_right_jk, fwhm_right_jk, A_right_jk = [], [], []

    for i in range(n_blocks):
        mask_jk = np.ones(N, dtype=bool)
        start, end = i*block_size, (i+1)*block_size
        mask_jk[start:end] = False

        data_jk = data[mask_jk]
        w_jk = None if weights is None else weights[mask_jk]
        if w_jk is not None:
            w_jk = w_jk / np.sum(w_jk)

        counts_jk, _ = np.histogram(data_jk, bins=bins, weights=w_jk)

        # Fit left
        popt, A, mu, fwhm, _ = fit_gaussian(left_mask, "Left JK", counts_jk)
        if not np.isnan(mu):
            mu_left_jk.append(mu)
            fwhm_left_jk.append(fwhm)
            A_left_jk.append(A)
        # Fit right
        popt, A, mu, fwhm, _ = fit_gaussian(right_mask, "Right JK", counts_jk)
        if not np.isnan(mu):
            mu_right_jk.append(mu)
            fwhm_right_jk.append(fwhm)
            A_right_jk.append(A)

    # Convert to arrays
    mu_left_jk = np.array(mu_left_jk); fwhm_left_jk = np.array(fwhm_left_jk); A_left_jk = np.array(A_left_jk)
    mu_right_jk = np.array(mu_right_jk); fwhm_right_jk = np.array(fwhm_right_jk); A_right_jk = np.array(A_right_jk)

    # Jackknife errors
    def jk_err(array): return np.sqrt((len(array)-1)/len(array) * np.sum((array - np.mean(array))**2))

    mu_left_err, fwhm_left_err, A_left_err = jk_err(mu_left_jk), jk_err(fwhm_left_jk), jk_err(A_left_jk)
    mu_right_err, fwhm_right_err, A_right_err = jk_err(mu_right_jk), jk_err(fwhm_right_jk), jk_err(A_right_jk)

    print("Left half:")
    print(f"{A_left:.5f},{mu_left:.5f},{fwhm_left:.5f},{A_left_err:.5f},{mu_left_err:.5f},{fwhm_left_err:.5f}")
    print("Right half:")
    print(f"{A_right:.5f},{mu_right:.5f},{fwhm_right:.5f},{A_right_err:.5f},{mu_right_err:.5f},{fwhm_right_err:.5f}")

    # --- Plot histogram and Gaussian fits ---
    plt.bar(bin_centers, counts, width=width, alpha=0.6, edgecolor='black', label='Histogram')
    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    if not np.isnan(mu_left):
        plt.plot(x_fit, gaussian(x_fit, *popt_left), 'r--', label='Left Gaussian fit')
        plt.axvline(mu_left, color='red', linestyle=':', label='Left mu')
    if not np.isnan(mu_right):
        plt.plot(x_fit, gaussian(x_fit, *popt_right), 'g--', label='Right Gaussian fit')
        plt.axvline(mu_right, color='green', linestyle=':', label='Right mu')
    plt.xlabel("Plaquette")
    plt.ylabel("Counts" if weights is None else "Weighted counts")
    plt.legend()
    plt.show()

    # Optional: plot halves
    if show_halves:
        plt.bar(bin_centers[left_mask], counts[left_mask], width=width, alpha=0.6, color='red', edgecolor='black')
        plt.title("Left half histogram")
        plt.show()
        plt.bar(bin_centers[right_mask], counts[right_mask], width=width, alpha=0.6, color='green', edgecolor='black')
        plt.title("Right half histogram")
        plt.show()

    return (A_left, A_left_err, mu_left, mu_left_err, fwhm_left, fwhm_left_err), \
           (A_right, A_right_err, mu_right, mu_right_err, fwhm_right, fwhm_right_err)

def w_array(delta_beta,SIZE):
    volume=SIZE**4*6
    logw_array = delta_beta * volume * phase_trans_raw
    maxlogw = np.max(logw_array)
    w_array = np.exp(np.clip(logw_array - maxlogw, -700, 700))
    wsum=np.sum(w_array)  # prevent overflow
    return w_array/wsum

bins=100

size3=pd.read_csv('/home/alexa/Z2_Lattice/Z2_Lattice_PT/PT3_RawMC.csv')
phase_trans_raw=size3['0.431000'].to_numpy()
delta_beta=0.431000-.4309904638155979
warray=w_array(delta_beta,3)
hist_plot(phase_trans_raw,bins=bins, weights=warray,threshold=0.7,n_blocks=50, show_halves=False)


size4=pd.read_csv('/home/alexa/Z2_Lattice/Z2_Lattice_PT/PT4_RawMC (4).csv')
phase_trans_raw=size4['0.437'].to_numpy()
delta_beta=0.4374132459696464-.437
warray=w_array(delta_beta,4)

hist_plot(phase_trans_raw, bins=bins, weights=warray, threshold=0.7, show_halves=False)

size5=pd.read_csv('/home/alexa/Z2_Lattice/Z2_Lattice_PT/PT5_RawMC (8).csv')
phase_trans_raw=size5['0.43915'].to_numpy()
delta_beta=0.4391895995649465-0.43915
warray=w_array(delta_beta,5)  # prevent overflow

hist_plot(phase_trans_raw, bins=bins, weights=warray, threshold=0.7, show_halves=False)

