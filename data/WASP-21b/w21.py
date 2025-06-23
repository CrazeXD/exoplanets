import numpy as np
import matplotlib.pyplot as plt

def load_spectrum(file):
    data = np.loadtxt(file)
    return data[:, 0], data[:, 1], data[:, 2]  # Wavelength, Flux, Error

files = [
    ("wasp21_HARPS_n1_full.txt", "wasp21_HARPS_n1_bin.txt", "HARPS Night 1"),
    ("wasp21_HARPS_n2_full.txt", "wasp21_HARPS_n2_bin.txt", "HARPS Night 2"),
    ("wasp21_HARPSN_n3_full.txt", "wasp21_HARPSN_n3_bin.txt", "HARPS Night 3")
]

vertical_lines = [5889.95, 5895.92]  # Example: Na I D lines

fig, axes = plt.subplots(len(files), 1, figsize=(10, 10), sharex=True)

for ax, (full_file, bin_file, label) in zip(axes, files):
    # Load full-resolution and binned data
    wavelength, flux, error = load_spectrum(full_file)
    bin_wavelength, bin_flux, bin_error = load_spectrum(bin_file)
    
    # Plot full-resolution data
    ax.errorbar(wavelength, flux, yerr=error, fmt='-', alpha=0.25, color = 'gray')#, label="Full Data")
    
    # Plot binned data
    ax.errorbar(bin_wavelength, bin_flux, yerr=bin_error, fmt='o', 
            markersize=3, label="HARPS 3.6-m", color='blue', 
            alpha=0.7, elinewidth=1, capsize=2, zorder=3)

    # Add vertical lines
    for vline in vertical_lines:
        ax.axvline(vline, color='orange', linestyle='--', alpha=0.8)

    ax.set_ylabel("Flux [%]")
    ax.set_title(label)
    ax.legend()
    plt.minorticks_on()
    ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)

	#axes[-1].set_xlabel("Wavelength [Å]")

plt.suptitle("WASP-21 Night-to-Night Transmission Spectra")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.tick_params(axis='both', direction='in', which='both', top=True, right=True)
#plt.minorticks_on()
# Save the figure
plt.savefig("wasp21_spectrum.png", dpi=300)
plt.savefig("wasp21_spectrum.pdf")

plt.show()
