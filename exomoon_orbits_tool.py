'''
Exomoon orbits tool
Author: Moritz Meyer zu Westram 
Contributor: Apurva V Oza
'''

import numpy as np
import pandas as pd
import rebound
import os
import matplotlib.pyplot as plt
import matplotlib.colors as col
import hsluv
import scipy.special
from matplotlib.widgets import Slider
from datetime import datetime, timezone
from astropy.time import Time
from astropy.utils.iers import conf
conf.auto_download = False

from barycorr.utc_tdb import JDUTC_to_BJDTDB


# Manually update leap seconds based on the current value
current_leap_seconds = 37  # Replace with the actual current leap seconds
current_utc_datetime = datetime.now(timezone.utc)  # Get the current UTC datetime
current_jd_utc = pd.Timestamp(current_utc_datetime).to_julian_date()
jd_tt = current_jd_utc + 32.184 + current_leap_seconds



def hours_to_deg(hours, minutes, seconds):
    return hours * 15 + minutes * 1/4 + seconds * 1/240


def arc_to_deg(degrees, arcminutes, arcseconds):
    # Be aware of the signs of the inputs!
    return degrees + arcminutes/60 + arcseconds/3600


RA = hours_to_deg(6, 4, 21.47346)   # equals 91.0895°
DEC = arc_to_deg(-16, -57, -55.1088)                      # equals -16.96531°

G = 6.6743e-11

M_PLANET = 6.92088630254e26 # 1.898e27
R_PLANET = 72278371.24956 # 69911e3


def make_anglemap(xmax, N=256, use_hpl=True, colorhue=38, saturation=90, flip=True):
    #h = np.ones(N) # hue
    #h[:N//2] = 11.6 # red
    #h[N//2:] = 258.6 # blue
    h = np.repeat(colorhue, N)    # hue
    s = saturation # saturation

    l = np.linspace(50, 100, N//2) # luminosity
    l = np.hstack((l, l[::-1]))

    colorlist = np.zeros((N,3))
    for ii in range(N):
        if use_hpl:
            colorlist[ii,:] = hsluv.hpluv_to_rgb((h[ii], s, l[ii]))
        else:
            colorlist[ii,:] = hsluv.hsluv_to_rgb((h[ii], s, l[ii]))
    colorlist[colorlist > 1] = 1 # correct numeric errors
    colorlist[colorlist < 0] = 0

    hpluv_anglemap = col.ListedColormap(colorlist)
    x = np.linspace(-xmax, xmax, 256)
    y = np.linspace(-xmax, xmax, 256)
    z = np.zeros((len(y), len(x)))  # make cartesian grid
    ymod = .1 * y / xmax
    xmod = .1 * x / xmax
    for ii in range(len(y)):
        z[ii] = np.arctan2(ymod[ii], xmod)  # simple angular function
        z[ii] = np.angle(scipy.special.gamma(xmod + 1j * ymod[ii]))  # some complex function

    if flip:
        z = np.flip(z, axis=1)

    return x, y, z, hpluv_anglemap


class TimingModel:

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            assert type(args[0][0]) == str, "Invalid input."
            self.fix_phase = args[0][1]
            self.fix_date = args[0][0]
            self.fix_date_bjd = self.get_bjd(args[0][0])
        elif len(args) == 2:
            assert type(args[0]) == str, "Invalid date input. Did you switch date and phase?"
            self.fix_phase = args[1]
            self.fix_date = args[0]
            self.fix_date_bjd = self.get_bjd(args[0])
        elif "fix_phase" in kwargs and "fix_date" in kwargs:
            self.fix_phase = kwargs['fix_phase']
            self.fix_date = kwargs['fix_date']
            self.fix_date_bjd = self.get_bjd(kwargs['fix_date'])
        else:
            raise ValueError("Invalid inputs.")

        self.ref_dates = []
        self.ref_dates_bjd = []
        self.ref_phases = []

    def set_approximate_phases(self, dates, phases):
        dates = np.atleast_1d(dates)
        phases = np.atleast_1d(phases)
        self.ref_dates = np.concatenate(([self.fix_date], dates))
        self.ref_dates_bjd = np.concatenate((self.fix_date_bjd, self.get_bjd(dates)))
        self.ref_phases = np.concatenate(([self.fix_phase], phases))
        assert len(self.ref_dates) == len(self.ref_phases), "You need to pass one phase to each date."

    @staticmethod
    def get_bjd(dates):
        if not isinstance(dates, (list, np.ndarray)):
            dates = [dates]
        date_bjd = []
        for date in dates:
            assert len(date) == 19, "Invalid input date-string encountered. Please enter as 'YYYY-MM-DD HH:MM:SS"
            dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
            ts = pd.Timestamp(year=dt.year, month=dt.month, day=dt.day,
                              hour=dt.hour, minute=dt.minute, second=dt.second)
            #date_bjd.append(barycorrPy.utc2bjd(ts.to_julian_date(), RA, DEC))

            JDUTC = Time(ts.to_julian_date(), format='jd', scale='utc')
            date_bjd.append(JDUTC_to_BJDTDB(JDUTC, ra=RA, dec=DEC)[0][0])

        return np.array(date_bjd)

    def semimajoraxis_sliderplot(self, dates, mark_orbitsolutions=False):
        dates = np.atleast_1d(dates)
        dates_bjd = self.get_bjd(dates)

        time_diff_seconds = np.array([(bjd - self.fix_date_bjd) * 86400 for bjd in dates_bjd])

        sort = dates_bjd.argsort()
        dates = dates[sort]
        time_diff_seconds = time_diff_seconds[sort]

        a_max = 1.6
        a_min = 1.2

        # Create a figure and axis objects for the subplots
        num_subplots = len(dates)
        fig, axes = plt.subplots(1, num_subplots, figsize=(20, 5))  # You can adjust the figsize as needed

        # Colormap background
        x, y, z, hpluv_anglemap = make_anglemap(1.2 * a_max * R_PLANET, colorhue=38, use_hpl=True)

        #colors1 = plt.cm.viridis(np.linspace(0., 1, 128))
        #colors2 = plt.cm.viridis_r(np.linspace(0., 1, 128))
        #colors = np.vstack((colors1, colors2))
        #hpluv_anglemap = col.LinearSegmentedColormap.from_list('my_colormap', colors)

        # TEST
        #x_p, y_p, z_p, hpluv_anglemap_planet = make_anglemap(R_PLANET, colorhue=259, saturation=100, flip=False,
        #                                                     use_hpl=True)
        #theta = np.linspace(0, 2 * np.pi, 400)
        #verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        #circle = matplotlib.path.Path(verts * R_PLANET)

        # Create a slider axes and slider
        slider_ax = plt.axes([0.95, 0.15, 0.03, 0.7])  # [left, bottom, width, height]

        if mark_orbitsolutions:
            if not os.path.isfile("timings.xlsx"):
                self.search_orbitsolution(error_threshold=0.001)
            df_solutions = pd.read_excel("timings.xlsx")
            a = df_solutions[df_solutions.columns[0]].to_numpy()
            assert len(a) > 0, "No solutions were found that could be marked. Check if 'timings.xlsx' is empty."
            for axis in a:
                slider_ax.axhline(y=axis, c='r', xmin=.2, xmax=.8)
            slider = Slider(slider_ax, r'$a_s$ [$R_P$]', a_min, a_max, valinit=1.0, orientation='vertical',
                            valstep=a, handle_style={'size': 0}, color=hsluv.hpluv_to_rgb((38, 90, 75)))

            # Handle ticks
            slider_ax.add_artist(slider_ax.yaxis)
            result = []
            current_range = []
            for num in a:
                if not current_range:
                    current_range.append(num)
                else:
                    diff = num - current_range[-1]
                    if diff <= 0.08:
                        current_range.append(num)
                    else:
                        if len(current_range) == 1:
                            result.append(current_range[0])
                        elif len(current_range) == 2:
                            mean_value = sum(current_range) / len(current_range)
                            result.append(mean_value)
                        else:
                            result.extend([current_range[0], current_range[-1]])
                        current_range = [num]
            if len(current_range) == 1:
                result.append(current_range[0])
            elif len(current_range) == 2:
                mean_value = sum(current_range) / len(current_range)
                result.append(mean_value)
            else:
                result.extend([current_range[0], current_range[-1]])
            slider_ax.set_yticks(result)

        else:
            slider = Slider(slider_ax, 'Semi-major axis', a_min, a_max, valinit=1.0, orientation='vertical')

        sim = rebound.Simulation()
        sim.add(m=M_PLANET)
        for ax in axes:
            ax.set_aspect("equal")
            planet = plt.Circle((0, 0), R_PLANET, zorder=2, color="royalblue")
            ax.add_patch(planet)
            ax.pcolormesh(x, y, z / np.pi, cmap=hpluv_anglemap, vmin=-1, vmax=1)
            #ax.pcolormesh(x_p, y_p, z_p / np.pi, cmap=hpluv_anglemap_planet, vmin=-1, vmax=1, clip_path=(circle, ax.transData))
            ax.set_xlim([-1.2 * a_max * R_PLANET, 1.2 * a_max * R_PLANET])
            ax.set_ylim([-1.2 * a_max * R_PLANET, 1.2 * a_max * R_PLANET])

        # Function to update the subplots
        def update(val):
            a = slider.val * R_PLANET
            T = 2 * np.pi * np.sqrt(a ** 3 / (G * M_PLANET))
            mean_anomalies = ((2 * np.pi / T * time_diff_seconds + self.fix_phase * 2 * np.pi) % (2 * np.pi)) / (2 * np.pi)
            for i, ax in enumerate(axes):
                sim.add(a=a, M=mean_anomalies[i] * 2 * np.pi)
                ax.clear()
                ax.pcolormesh(x, y, z / np.pi, cmap=hpluv_anglemap, vmin=-1, vmax=1)
                op = rebound.OrbitPlot(sim, fig=fig, ax=ax, show_primary=False, color='red')
                op.particles.set_color(["red"])
                planet = plt.Circle((0, 0), R_PLANET, zorder=2, color="royalblue")
                ax.add_patch(planet)

                # Anchor position for text
                text_x = 0.6 * a_max * R_PLANET - 0.03 * a_max * R_PLANET * len(str(np.around(mean_anomalies[i], 2)))
                text_y = 0.8 * a_max * R_PLANET
                ax.text(text_x, text_y, f'{float(np.around(mean_anomalies[i] * 360, 2))}',
                        bbox={'facecolor': 'peachpuff', 'alpha': 0.8, 'pad': 7})

                ax.set_title(f'{dates[i][:10]} | {dates[i][11:]}', pad=18)
                #ax.set_xlim([-1.2 * a_max * R_PLANET, 1.2 * a_max * R_PLANET])
                #ax.set_ylim([-1.2 * a_max * R_PLANET, 1.2 * a_max * R_PLANET])
                sim.remove(1)

            plt.savefig("test.png")

        # Link the slider to the update function
        slider.on_changed(update)

        # Function to handle keyboard events for the slider
        def on_key(event):
            step = 0.01  # You can adjust the step size as needed
            if event.key == 'up':
                slider.set_val(min(slider.val + step, a_max))
            elif event.key == 'down':
                slider.set_val(max(slider.val - step, a_min))
            fig.canvas.draw_idle()

        # Connect the keyboard event handler
        fig.canvas.mpl_connect('key_press_event', on_key)

        # Display the plot
        plt.show()

    def search_orbitsolution(self, additional_dates=None, a_min=1.1, a_max=1.6, error_threshold=0.01):
        assert len(self.ref_dates) > 1, "Please set at least one additional known/approx. date and phase."
        a = R_PLANET * np.linspace(a_min, a_max, 20000)
        T = 2 * np.pi * np.sqrt(a ** 3 / (G * M_PLANET))

        ref_dates_timediff = np.array([(bjd - self.fix_date_bjd[0]) * 86400 for bjd in self.ref_dates_bjd])

        if additional_dates is not None:
            additional_dates = np.atleast_1d(additional_dates)
            add_dates_bjd = self.get_bjd(additional_dates)
            add_dates_timediff = np.array([(bjd - self.fix_date_bjd[0]) * 86400 for bjd in add_dates_bjd])
            dates_timediff = np.concatenate((ref_dates_timediff, add_dates_timediff))
            dates = np.concatenate((self.ref_dates, additional_dates))
        else:
            dates_timediff = ref_dates_timediff
            dates = self.ref_dates

        mean_anomalies = ((2 * np.pi / T[:, np.newaxis] * dates_timediff + self.fix_phase * 2 * np.pi)
                          % (2 * np.pi)) / (2 * np.pi)

        phase_differences = np.abs(mean_anomalies[:, :len(self.ref_phases)] - self.ref_phases)
        threshold_mask = np.all(phase_differences <= error_threshold, axis=1)

        solutions_anomaly = np.round(mean_anomalies[threshold_mask], 4)
        solutions_a = a[threshold_mask] / R_PLANET
        solutions_T = T[threshold_mask] / 3600
        solutions_concat = np.vstack((solutions_a, solutions_T, solutions_anomaly.T)).T

        df_columns = ['a [R_P]', 'T [hr]'] + [f'Phase of date {dates[i]}' for i in range(len(dates_timediff))]
        df = pd.DataFrame(solutions_concat, columns=df_columns)
        df.to_csv('timings.csv', index=False)

#2024-12-03, 2025-02-19
if __name__ == "__main__":
 #   dates = ["2015-12-07 04:31:00", "2016-01-01 05:22:00", "2019-11-11 13:32:00", "2020-12-16 03:14:00", "2024-02-05 05:03:00"]
#    dates = ["2015-12-07 04:31:00", "2016-01-01 05:22:00", "2019-11-11 13:32:00", "2020-12-16 03:14:00", "2024-11-20 12:16:00"]
    dates = ["2015-12-07 04:31:00", "2016-01-01 05:22:00", "2019-11-11 13:32:00", "2020-12-16 03:14:00", "2024-12-04 10:04:00", "2025-02-20 07:24:00", "2025-11-25 11:34:00", "2026-01-17 08:02:00"]
    fix_date_phase = ("2019-11-11 13:32:00", 0.0)   # This will be fixed directly. # Might be slightly larger phase.
    tm = TimingModel(fix_date_phase) 
    tm.set_approximate_phases("2016-01-15 03:11:00", 0.1)
    tm.set_approximate_phases("2016-01-01 05:22:00", 0.7)
    tm.set_approximate_phases("2020-12-16 03:11:00", 0.1) #0.6
    tm.search_orbitsolution(additional_dates=dates, error_threshold=3e-5)   # Low error_threshold fixes the phases passed in previous line.
    tm.semimajoraxis_sliderplot(dates + ["2016-01-15 03:11:00"], mark_orbitsolutions=True) # Theory counters to the fix_date_phase
# "2021-01-07 09:21:00"