import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

# These imports assume the Prometheus package is in the Python path
# or in the specified relative directory.
import Prometheus.pythonScripts.celestialBodies as bodies
import Prometheus.pythonScripts.constants as const
import Prometheus.pythonScripts.gasProperties as gasprop
import Prometheus.pythonScripts.geometryHandler as geom


def get_exomoon_params(
    planet_name: str,
    element_name: str,
    num_particles: float,
    moon_radius_io: float,
    moon_orbit_planet_radii: float,
    moon_q: float,
    clv_u1: float,
    clv_u2: float,
    light_curve: bool = False,
    observed_data_phases: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Sets up parameters for a generic planet with an exomoon scenario.

    Args:
        planet_name: The name of the planet to simulate (must exist in AvailablePlanets).
        element_name: The name of the atomic species in the exosphere.
        num_particles: Total number of particles in the exosphere.
        moon_radius_io: Radius of the moon in units of Io's radius.
        moon_orbit_planet_radii: Orbital distance of the moon in units of the host planet's radius.
        moon_q: Density profile parameter for the moon's exosphere.
        clv_u1: Linear limb-darkening coefficient for the host star.
        clv_u2: Quadratic limb-darkening coefficient for the host star.
        light_curve: If True, configure for light curve simulation.
        observed_data_phases: Optional array of observed data phases to set simulation range.

    Returns:
        A dictionary of parameters for the PROMETHEUS simulation.
    """
    planet_obj = bodies.AvailablePlanets().findPlanet(planet_name)
    if planet_obj is None:
        raise ValueError(
            f"{planet_name} not found in AvailablePlanets. Check celestialBodies.py.")

    params: Dict[str, Any] = {
        "Fundamentals": {
            "ExomoonSource": True,
            "DopplerPlanetRotation": False,
            "CLV_variations": True,
            "RM_effect": False,
            "DopplerOrbitalMotion": True,
        },
        "Architecture": {
            "planetName": planet_name,
            "CLV_u1": clv_u1,
            "CLV_u2": clv_u2,
            "R_moon": moon_radius_io * const.R_Io,
            "a_moon": moon_orbit_planet_radii * planet_obj.R,
            # Position moon for transit (in front of planet). May need adjustment.
            "starting_orbphase_moon": np.pi,
        },
        "Scenarios": {
            "exomoon": {"q_moon": moon_q}
        },
        "Species": {"exomoon": {element_name: {"Nparticles": num_particles}}},
        "Grids": {
            "x_midpoint": planet_obj.a,
            "x_border": planet_obj.hostStar.R,
            "x_steps": 30,
            "phi_steps": 60,
            "rho_steps": 40,
            "upper_rho": planet_obj.hostStar.R,
            "orbphase_border": 0.0,
            "orbphase_steps": 1,
        },
    }

    # The simulation grid expects orbphase_border in RADIANS.
    if light_curve and observed_data_phases is not None and len(observed_data_phases) > 0:
        min_obs_phase = np.min(observed_data_phases)
        max_obs_phase = np.max(observed_data_phases)
        # Calculate the border in fractional phase first
        phase_border_fractional = max(
            abs(min_obs_phase), abs(max_obs_phase)) + 0.005
        # Convert fractional phase to radians for the simulation
        params["Grids"]["orbphase_border"] = phase_border_fractional * 2.0 * np.pi
        # Number of points for the simulated light curve
        params["Grids"]["orbphase_steps"] = 201
    elif light_curve:
        phase_border_fractional = 0.065  # Default if no obs data for range
        params["Grids"]["orbphase_border"] = phase_border_fractional * 2.0 * np.pi
        params["Grids"]["orbphase_steps"] = 201

    if element_name == "NaI":
        mass_Na = const.AvailableSpecies().findSpecies("NaI").mass
        sigma_v_Na = (
            # Use host star's effective temperature for thermal velocity dispersion
            const.k_B * planet_obj.hostStar.T_eff
            / mass_Na
        ) ** 0.5
        params["Species"]["exomoon"][element_name]["sigma_v"] = sigma_v_Na
        grids_Na = {
            "lower_w": 5.888e-05,
            "upper_w": 5.900e-05,
            "resolutionLow": 5e-09,
            "widthHighRes": 0.75e-8,
            "resolutionHigh": 2e-10,
        }
        params["Grids"].update(grids_Na)
    elif element_name == "KI":
        params["Species"]["exomoon"][element_name]["sigma_v"] = 1e5
        grids_K = {
            "lower_w": 7.660e-05,
            "upper_w": 7.705e-05,
            "resolutionLow": 5e-09,
            "widthHighRes": 1e-08,
            "resolutionHigh": 2e-10,
        }
        params["Grids"].update(grids_K)
    else:
        print(
            f"Warning: Element {element_name} specific grid/species params not fully set.")
        params["Grids"]["lower_w"] = 0.5e-4
        params["Grids"]["upper_w"] = 1.0e-4
        params["Grids"]["resolutionLow"] = 1e-8
        params["Grids"]["widthHighRes"] = 1e-8
        params["Grids"]["resolutionHigh"] = 1e-9
    return params


def load_observed_light_curve(filepath: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        data = np.loadtxt(filepath)
        if data.ndim == 1:  # Handle case where only one data point might be in the file
            if len(data) == 3:
                return np.array([data[0]]), np.array([data[1]]), np.array([data[2]])
            elif len(data) == 2:
                # No error
                return np.array([data[0]]), np.array([data[1]]), np.array([0.0])
            else:
                raise ValueError(
                    f"Single row in data file {filepath} does not have 2 or 3 columns.")

        if data.shape[1] == 3:
            return data[:, 0], data[:, 1], data[:, 2]
        elif data.shape[1] == 2:
            print(
                f"Warning: Data file {filepath} has 2 columns. Assuming no error bars provided.")
            return data[:, 0], data[:, 1], np.zeros_like(data[:, 0])
        else:
            raise ValueError(f"Data file {filepath} must have 2 or 3 columns.")
    except Exception as e:
        print(f"Error loading observed data from {filepath}: {e}")
        return None, None, None


def plot_light_curve(
    orbital_phases_sim: np.ndarray,
    light_curve_values_sim: np.ndarray,
    orbital_phases_obs: Optional[np.ndarray],
    light_curve_values_obs: Optional[np.ndarray],
    light_curve_errors_obs: Optional[np.ndarray],
    element_name: str,
    planet_name: str,
    num_particles_str: str,
    filter_bandwidth_angstrom: float,
    planet_object: bodies.Planet
) -> None:
    plt.rcParams['font.family'] = "STIXGeneral"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6

    fig, ax = plt.subplots(figsize=(10, 6))

    # The observed data from the file is already relative flux (i.e. Flux_obs - 1)
    if orbital_phases_obs is not None and light_curve_values_obs is not None:
        ax.errorbar(orbital_phases_obs, light_curve_values_obs, yerr=light_curve_errors_obs, fmt='o', color='tab:blue',
                    label='Observed Data',
                    capsize=4, elinewidth=1.5, markeredgewidth=1.5, markersize=7, ecolor='dimgray', zorder=5)

    simulated_relative_flux_change = light_curve_values_sim - 1.0
    ax.plot(orbital_phases_sim, simulated_relative_flux_change, color="orangered",
            label=f"Simulated {element_name} Light Curve", zorder=10)

    title_text = (
        f"Exomoon {element_name} Light Curve for {planet_name}\n"
        f"Particles: {num_particles_str}, Filter: {filter_bandwidth_angstrom:.2f} Å"
    )
    ax.set_title(title_text, fontweight='bold')
    ax.set_xlabel("Orbital Phase (𝜙)")
    ax.set_ylabel("Relative Flux Change (F/F_star - 1)")

    # Determine y-limits
    all_y_min: List[float] = []
    all_y_max: List[float] = []
    if light_curve_values_obs is not None and light_curve_errors_obs is not None:
        all_y_min.append(
            np.min(light_curve_values_obs - light_curve_errors_obs))
        all_y_max.append(
            np.max(light_curve_values_obs + light_curve_errors_obs))
    if len(simulated_relative_flux_change) > 0:
        all_y_min.append(np.min(simulated_relative_flux_change))
        all_y_max.append(np.max(simulated_relative_flux_change))

    if all_y_min and all_y_max:
        min_y_data = min(all_y_min)
        max_y_data = max(all_y_max)
        padding = 0.1 * (max_y_data - min_y_data) if (max_y_data -
                                                      min_y_data) > 1e-6 else 0.005
        ax.set_ylim(min_y_data - padding, max_y_data + padding)
    else:  # Fallback if no data to determine limits
        ax.set_ylim(-0.025, 0.025)

    # Determine x-limits
    all_x_min: List[float] = []
    all_x_max: List[float] = []
    if orbital_phases_obs is not None:
        all_x_min.append(np.min(orbital_phases_obs))
        all_x_max.append(np.max(orbital_phases_obs))
    if len(orbital_phases_sim) > 0:
        all_x_min.append(np.min(orbital_phases_sim))
        all_x_max.append(np.max(orbital_phases_sim))

    if all_x_min and all_x_max:
        ax.set_xlim(min(all_x_min) - 0.005, max(all_x_max) + 0.005)
    else:  # Fallback
        ax.set_xlim(-0.05, 0.07)

    transit_duration_hr: float = planet_object.transitDuration
    transit_duration_days: float = transit_duration_hr / 24.0
    planet_orbital_period_days: float = planet_object.orbitalPeriod

    transit_phase_duration_half: float = (
        transit_duration_days / planet_orbital_period_days) / 2.0
    ax.axvline(-transit_phase_duration_half, color='k', linestyle='--', linewidth=2,
               label=f'Planet Transit Start/End (±{transit_phase_duration_half:.3f})')
    ax.axvline(transit_phase_duration_half,
               color='k', linestyle='--', linewidth=2)

    ax.axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.7)

    # Update legend to avoid duplicate labels for axvline
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')

    ax.xaxis.set_major_locator(MaxNLocator(
        nbins=8, prune='both'))  # Prune to avoid crowding
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))

    fig.tight_layout(pad=1.5)
    plt.show()


def plot_transmission_spectrum(
    wavelength_cm: np.ndarray,
    spectrum_R: np.ndarray,
    element_name: str,
    planet_name: str,
    num_particles_str: str,
) -> None:
    wavelength_angstroms = wavelength_cm * 1e8
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(wavelength_angstroms, spectrum_R,
            linewidth=1.5, color="dodgerblue")

    title_text = (
        f"Exomoon {element_name} Transmission Spectrum for {planet_name}\n"
        f"Total Particles: {num_particles_str}"
    )
    ax.set_title(title_text, fontsize=16, fontweight='bold')
    ax.set_xlabel("Wavelength (Angstrom)", fontsize=14)
    ax.set_ylabel("Relative Flux (F_out / F_in)", fontsize=14)

    min_R_val = np.min(spectrum_R) if len(spectrum_R) > 0 else 1.0
    max_R_val = np.max(spectrum_R) if len(spectrum_R) > 0 else 1.0

    padding = 0.05 * (max_R_val - min_R_val) if (max_R_val -
                                                 min_R_val) > 1e-6 else 0.001
    plot_bottom = min_R_val - padding
    plot_top = max_R_val + padding

    if np.allclose(min_R_val, 1.0) and np.allclose(max_R_val, 1.0):
        plot_bottom = 0.995
        plot_top = 1.005

    ax.set_ylim(bottom=plot_bottom, top=plot_top)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.4f}"))
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    plt.show()


# --- Main execution ---
if __name__ == "__main__":
    # ######################################################################### #
    # ################        USER CONFIGURATION BLOCK         ################ #
    # ######################################################################### #

    # --- Planet & Star Parameters ---
    # Name of the planet (must be defined in Prometheus/pythonScripts/celestialBodies.py)
    PLANET_NAME = input("Enter the planet name (e.g., WASP-49b): ").strip()
    # Quadratic limb-darkening coefficients for the host star
    CLV_U1 = float(input("Enter the CLV u1 coefficient (e.g., 0.22): ").strip())
    CLV_U2 = float(input("Enter the CLV u2 coefficient (e.g., 0.34): ").strip())

    # --- Exomoon & Exosphere Parameters ---
    ELEMENT_NAME = input("Enter the name of the species in the exosphere (e.g. NaI, KI): ")  # Species in the exosphere (e.g., "NaI", "KI")
    NUM_PARTICLES = float(input("Enter the total number of particles in the cloud: ")) # Total number of particles in the cloud
    MOON_RADIUS_IO = 1.0   # Radius of the moon in units of Io's radius
    MOON_ORBIT_PLANET_RADII = float(input("Enter the moon's orbital distance in units of the planet's radius: ")) # Moon's orbital distance in units of the planet's radius
    MOON_EXOSPHERE_Q = 3.34 # Exosphere density fall-off parameter (q from Chamberlain model)

    # --- Simulation Mode & Data ---
    LIGHT_CURVE_MODE = {"LC": True, "T": False}.get(input("Set up in lightcurve or transmission spectra mode (LC, T)? ")) # True for light curve, False for single transmission spectrum
    FILTER_BANDWIDTH_ANGSTROM = 0.75 # Passband filter width for light curve in Angstroms
    # Path to observed data file (2 or 3 columns: phase, flux, [optional] error)
    if LIGHT_CURVE_MODE:
        OBSERVED_DATA_FILEPATH = input("Enter the path to the observed data file: ").strip()
        FILTER_BANDWIDTH_CM = FILTER_BANDWIDTH_ANGSTROM * 1e-8

        obs_phases, obs_flux, obs_errors = load_observed_light_curve(
            OBSERVED_DATA_FILEPATH)
        
        if obs_phases is None:
            print("Could not load observed data. Running simulation without comparison data overlay.")

    # ######################################################################### #
    # #####################        END CONFIGURATION         #################### #
    # ######################################################################### #

    print(
        f"--- Simulation for {ELEMENT_NAME} around an exomoon of {PLANET_NAME} ---")
    print(f"Mode: {'Light Curve' if LIGHT_CURVE_MODE else 'Single Spectrum'}")
    print(f"Number of {ELEMENT_NAME} particles: {NUM_PARTICLES:.1e}")
    if LIGHT_CURVE_MODE:
        params = get_exomoon_params(
            PLANET_NAME, ELEMENT_NAME, NUM_PARTICLES, MOON_RADIUS_IO,
            MOON_ORBIT_PLANET_RADII, MOON_EXOSPHERE_Q, CLV_U1, CLV_U2,
            light_curve=LIGHT_CURVE_MODE, observed_data_phases=obs_phases
        )
        print(
            f"Calculated Sigma_v for {ELEMENT_NAME}: {params['Species']['exomoon'][ELEMENT_NAME].get('sigma_v', 'N/A'):.2e} cm/s")
        print(
            f"Orbital phase border for simulation: +/- {params['Grids']['orbphase_border'] / (2 * np.pi):.4f} (fractional), {params['Grids']['orbphase_border']:.4f} (radians)")
        print(
            f"Number of orbital phase steps: {params['Grids']['orbphase_steps']}")
    else:
        params = get_exomoon_params(
            PLANET_NAME, ELEMENT_NAME, NUM_PARTICLES, MOON_RADIUS_IO,
            MOON_ORBIT_PLANET_RADII, MOON_EXOSPHERE_Q, CLV_U1, CLV_U2,
            light_curve=LIGHT_CURVE_MODE
        )
        print(
            f"Calculated Sigma_v for {ELEMENT_NAME}: {params['Species']['exomoon'][ELEMENT_NAME].get('sigma_v', 'N/A'):.2e} cm/s")

    fundamentalsDict = params["Fundamentals"]
    scenarioDict = params["Scenarios"]
    architectureDict = params["Architecture"]
    speciesDict = params["Species"]
    gridsDict = params["Grids"]

    planet = bodies.AvailablePlanets().findPlanet(
        architectureDict["planetName"])
    if fundamentalsDict["CLV_variations"]:
        planet.hostStar.addCLVparameters(
            architectureDict.get(
                "CLV_u1", 0.0), architectureDict.get("CLV_u2", 0.0)
        )

    wavelengthGrid = gasprop.WavelengthGrid(
        gridsDict["lower_w"], gridsDict["upper_w"], gridsDict["widthHighRes"],
        gridsDict["resolutionLow"], gridsDict["resolutionHigh"]
    )
    spatialGrid = geom.Grid(
        gridsDict["x_midpoint"], gridsDict["x_border"], int(
            gridsDict["x_steps"]),
        gridsDict["upper_rho"], int(
            gridsDict["rho_steps"]), int(gridsDict["phi_steps"]),
        gridsDict["orbphase_border"], int(gridsDict["orbphase_steps"])
    )

    scenarioList = []
    for key_scenario in scenarioDict.keys():
        if key_scenario == "exomoon":
            moon = bodies.Moon(
                architectureDict["starting_orbphase_moon"],
                architectureDict["R_moon"], architectureDict["a_moon"], planet
            )
            current_species_name = list(speciesDict["exomoon"].keys())[0]
            num_particles_for_constructor = speciesDict["exomoon"][current_species_name]["Nparticles"]
            scenarioList.append(
                gasprop.MoonExosphere(
                    num_particles_for_constructor,
                    scenarioDict["exomoon"]["q_moon"], moon
                )
            )

    for idx, key_scenario in enumerate(scenarioDict.keys()):
        for key_species in speciesDict[key_scenario].keys():
            absorberDict = speciesDict[key_scenario][key_species]
            if key_species in const.AvailableSpecies().listSpeciesNames():
                if "sigma_v" in absorberDict:
                    scenarioList[idx].addConstituent(
                        key_species, absorberDict["sigma_v"])
                    scenarioList[idx].constituents[-1].addLookupFunctionToConstituent(
                        wavelengthGrid)
                elif "chi" in absorberDict:
                    scenarioList[idx].addConstituent(
                        key_species, absorberDict["chi"])
                    scenarioList[idx].constituents[-1].addLookupFunctionToConstituent(
                        wavelengthGrid)
                else:
                    raise ValueError(
                        f"Missing 'sigma_v' or 'chi' for atomic species {key_species}")
            else:  # Molecular
                if "T" in absorberDict:
                    scenarioList[idx].addMolecularConstituent(
                        key_species, absorberDict.get("chi", 1.0))
                    scenarioList[idx].constituents[-1].T = absorberDict["T"]
                    scenarioList[idx].constituents[-1].addLookupFunctionToConstituent()
                else:
                    raise ValueError(
                        f"Missing 'T' for molecular species {key_species}")

    atmos = gasprop.Atmosphere(
        scenarioList, fundamentalsDict["DopplerOrbitalMotion"])
    main_transit_model = gasprop.Transit(atmos, wavelengthGrid, spatialGrid)
    main_transit_model.addWavelength()

    print("\nRunning PROMETHEUS simulation...")
    start_time = datetime.now()
    R_values_all_wavelengths = main_transit_model.sumOverChords()
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"PROMETHEUS simulation finished. Elapsed time: {elapsed_time}")

    if LIGHT_CURVE_MODE:
        if R_values_all_wavelengths.ndim == 2 and R_values_all_wavelengths.shape[0] > 0 and R_values_all_wavelengths.shape[1] > 0:
            wavelengths_simulated_cm = main_transit_model.wavelength
            orbital_phases_simulated_rad = spatialGrid.constructOrbphaseAxis()
            orbital_phases_plot_sim = orbital_phases_simulated_rad / \
                (2.0 * np.pi)

            light_curve_simulated = []
            v_los_moon_total = np.zeros_like(orbital_phases_simulated_rad)
            if scenarioList and hasattr(scenarioList[0], 'hasMoon') and scenarioList[0].hasMoon:
                moon_obj = scenarioList[0].moon
                for i, orb_phase_rad in enumerate(orbital_phases_simulated_rad):
                    v_los_moon_total[i] = moon_obj.getLOSvelocity(
                        orb_phase_rad)
            else:
                v_los_planet_orb = planet.getLOSvelocity(
                    orbital_phases_simulated_rad)
                v_los_moon_total = v_los_planet_orb

            shifts = const.calculateDopplerShift(v_los_moon_total)
            NaD1_center_cm = 5897.558147e-8
            NaD2_center_cm = 5891.583253e-8

            for i in range(R_values_all_wavelengths.shape[0]):
                current_R_spectrum = R_values_all_wavelengths[i, :]
                current_shift = shifts[i]
                # CORRECTED: Doppler shift is applied by multiplying rest wavelength by shift factor.
                # The observed wavelength lambda_obs = lambda_rest * (1 + v/c).
                shifted_D1_center = NaD1_center_cm * current_shift
                shifted_D2_center = NaD2_center_cm * current_shift

                sel_filter_D1 = (
                    (wavelengths_simulated_cm >= shifted_D1_center - FILTER_BANDWIDTH_CM / 2.0) &
                    (wavelengths_simulated_cm <= shifted_D1_center + FILTER_BANDWIDTH_CM / 2.0)
                )
                sel_filter_D2 = (
                    (wavelengths_simulated_cm >= shifted_D2_center - FILTER_BANDWIDTH_CM / 2.0) &
                    (wavelengths_simulated_cm <= shifted_D2_center + FILTER_BANDWIDTH_CM / 2.0)
                )
                sel_filter_combined = sel_filter_D1 | sel_filter_D2

                # R_values is already the relative flux (F_out/F_in). We average it over the filter band.
                if np.any(sel_filter_combined):
                    mean_R_in_filter = np.mean(
                        current_R_spectrum[sel_filter_combined])
                else:
                    mean_R_in_filter = 1.0
                light_curve_simulated.append(mean_R_in_filter)
            light_curve_simulated = np.array(light_curve_simulated)

            plot_light_curve(
                orbital_phases_plot_sim, light_curve_simulated,
                obs_phases, obs_flux, obs_errors,
                ELEMENT_NAME, architectureDict["planetName"],
                f"{NUM_PARTICLES:.1e}", FILTER_BANDWIDTH_ANGSTROM,
                planet
            )
            output_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(output_dir, exist_ok=True)
            lc_filename = os.path.join(
                output_dir, f"{PLANET_NAME}_{ELEMENT_NAME}_exomoon_lightcurve_vs_obs.txt")
            np.savetxt(lc_filename, np.vstack((orbital_phases_plot_sim, light_curve_simulated-1.0)).T,
                       header="Orbital_Phase_(fraction) Relative_Flux_Change", fmt="% .8e")
            print(f"Simulated light curve data saved to: {lc_filename}")
        else:
            print(
                f"Warning: R_values_all_wavelengths has unexpected shape {R_values_all_wavelengths.shape}. Cannot plot light curve.")
    else:
        print("Single spectrum mode selected.")
        if R_values_all_wavelengths.ndim == 2 and R_values_all_wavelengths.shape[0] > 0 and R_values_all_wavelengths.shape[1] > 0:
            spectrum_to_plot_single = R_values_all_wavelengths[0, :].copy()
            continuum_single = np.max(spectrum_to_plot_single)
            if continuum_single > 1e-9 and not np.isclose(continuum_single, 1.0):
                spectrum_to_plot_single /= continuum_single

            plot_transmission_spectrum(
                main_transit_model.wavelength, spectrum_to_plot_single,
                ELEMENT_NAME, architectureDict["planetName"], f"{NUM_PARTICLES:.1e}"
            )
            output_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(output_dir, exist_ok=True)
            spectrum_filename = os.path.join(
                output_dir, f"{PLANET_NAME}_{ELEMENT_NAME}_exomoon_spectrum.txt")
            np.savetxt(spectrum_filename, np.vstack((main_transit_model.wavelength, spectrum_to_plot_single)).T,
                       header="Wavelength_cm Relative_Flux_R", fmt="% .8e")
            print(f"Simulated spectrum data saved to: {spectrum_filename}")
        else:
            print(
                f"Warning: R_values_all_wavelengths has unexpected shape {R_values_all_wavelengths.shape} for single spectrum.")
    print("--- End of Simulation ---")