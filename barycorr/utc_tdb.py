import numpy as np
import os
import math
import astropy
from astropy.coordinates import EarthLocation, SkyCoord     # requires 'pip install astropy'
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
try:
    from astroquery.simbad import Simbad                    # requires 'pip install astroquery'
except:
    print('Cannot import astroquery.simbad')
import astropy.units as u
import astropy.constants as ac
import barycorr.PINT_erfautils as PINT


fpath=os.path.join(os.path.dirname(__file__),'data')

# Parsing constants #
AU = ac.au.value # [m]
c = ac.c.value # Speed of light [m/s]
pctoau = 3600*180/np.pi # No. of AU in one parsec
SECS_PER_DAY = 3600 * 24
year = 365.25*SECS_PER_DAY # [s]
kmstoauyr = 1000 * year/AU #km/s to AU/year


def get_stellar_data(name=''):
    '''
    Function to query Simbad for following stellar information RA, Dec, PMRA, PMDec, Parallax Epoch
    INPUTS:
        name = Name of source. Example


    '''
    warning = []

    customSimbad = Simbad()
    customSimbad.add_votable_fields('ra(2;A;ICRS;J2000)', 'dec(2;D;ICRS;J2000)', 'pm', 'plx', 'parallax', 'rv_value')
    # Simbad.list_votable_fields()
    customSimbad.remove_votable_fields('coordinates')
    # Simbad.get_field_description('orv')
    obj = customSimbad.query_object(name)
    if obj is None:
        raise ValueError(
            'ERROR: {} target not found. Check target name or enter RA,Dec,PMRA,PMDec,Plx,RV,Epoch manually\n\n'.format(
                name))
    else:
        warning += ['{} queried from SIMBAD.'.format(name)]

    # Check for masked values
    if all([not x for x in [obj.mask[0][i] for i in obj.colnames]]) == False:
        warning += ['Masked values present in queried dataset']

    obj = obj.filled(None)

    pos = SkyCoord(ra=obj['RA_2_A_ICRS_J2000'], dec=obj['DEC_2_D_ICRS_J2000'], unit=(u.hourangle, u.deg))
    ra = pos.ra.value[0]
    dec = pos.dec.value[0]
    pmra = obj['PMRA'][0]
    pmdec = obj['PMDEC'][0]
    plx = obj['PLX_VALUE'][0]
    rv = obj['RV_VALUE'][0] * 1000  # SIMBAD output is in km/s. Converting to m/s
    epoch = 2451545.0

    star = {'ra': ra, 'dec': dec, 'pmra': pmra, 'pmdec': pmdec, 'px': plx, 'rv': rv, 'epoch': epoch}

    # Fill Masked values with None. Again.
    for i in star:
        if star[i] > 1e10:
            star[i] = None

    warning += ['Values queried from SIMBAD are {}'.format(star)]

    return star, warning


def find_hip(hip_index, cat_dir=os.path.join(*[os.path.dirname(__file__), 'data', 'hip2.dat'])):
    '''
    NOTE: Hipparcos Catalogue Epoch is J1991.25 or JD 2448349.0625

    INPUT:
        hip_index : The index of the star that needs to be searched.
        cat_dir : Directory where catalogue is saved.
    OUTPUT:
        hip_id : Index of Star
        ra : RA of star in degrees
        dec : Declination of star in degrees
        px_mas : Parallax angle in milli-arcseconds
        pmra : Proper motion in RA in milli-arcseconds/year
        pmdec : Proper motion in Declination in milli-arcseconds/year
        epoch : Epoch of Catalogue - J1991.25 , JD 2448349.0625

    '''
    epoch = 2448349.0625

    hip_id = []
    ra = []
    dec = []
    px_mas = []
    pmra = []
    pmdec = []

    with open(cat_dir) as f:
        for line in f:
            a = line.split()
            hip_id.append(float(a[0]))
            ra.append((float(a[4]) * 180.) / np.pi)  # Convert to degrees
            dec.append((float(a[5]) * 180.) / np.pi)  # Convert from radians to degrees
            px_mas.append(float(a[6]))  # in mas
            pmra.append(float(a[7]))  # in mas/year
            pmdec.append(float(a[8]))  # in mas/year

    index = np.where(np.array(hip_id) == hip_index)[0][0]

    star = {'ra': ra[index], 'dec': dec[index], 'pmra': pmra[index], 'pmdec': pmdec[index], 'px': px_mas[index],
            'epoch': epoch}

    return star


def JDUTC_to_BJDTDB(JDUTC,
       starname = '', hip_id=None, ra=None, dec=None, epoch=None, pmra=None, pmdec=None, px=None, rv=None,
       obsname='', lat=0., longi=0., alt=0.,
       ephemeris='de430', leap_dir=os.path.join(os.path.dirname(__file__),'data'), leap_update=True):

    '''
    Time conversion between JDUTC to BJDTDB. See Eastman et al. (2010)
    This code is precise to about 10 ms.

    Calling procedure for JDUTC_to_BJDTDB. Accepts vector time object (i.e., multiple observation JD values).

    INPUT:
        JDUTC : Can enter multiple times in Astropy Time object or as float. Will loop through and find barycentric velocity correction corresponding to those times.
                In UTC Scale. If using float, be careful about format and scale used.
        starname : Name of target. Will query SIMBAD database.
                                OR / AND
        hip_id : Hipparcos Catalog ID. (Integer) . Epoch will be taken to be Catalogue Epoch or J1991.25
                If specified then ra,dec,pmra,pmdec,px, and epoch need not be specified.
                                OR / AND
        ra, dec : RA and Dec of star [degrees].
        epoch : Epoch of coordinates in Julian Date. Default is J2000 or 2451545.
        pmra : Proper motion in RA [mas/year]. Eg. PMRA = d(RA)/dt * cos(dec). Default is 0.
        pmdec : Proper motion in Dec [mas/year]. Default is 0.
        px : Parallax of target [mas]. Default is 0.

        obsname : Name of Observatory as defined in Astropy EarthLocation routine. Can check list by EarthLocation.get_site_names().
                  If obsname is not used, then can enter lat,long,alt.
                                OR
        lat : Latitude of observatory in [degrees]. North (+ve) and South (-ve).
        longi : Longitude of observatory [degrees]. East (+ve) and West (-ve).
        alt : Altitude of observatory [m].

        rv : Radial Velocity of Target [m/s]. Default is 0.
        ephemeris : Name of Ephemeris to be used. List of Ephemeris as queried by jplephem. Default is DE430.
                    For first use Astropy will download the Ephemeris ( for DE430 ~100MB). Options for ephemeris inputs are
                    ['de432s','de430',
                    'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de423_for_mercury_and_venus/de423.bsp',
                    'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de405.bsp']
        leap_dir : Directory where leap seconds file will be saved and maintained (STRING). Eg. '/Users/abc/home/savehere/'. Default is script directory. [Not used with versions >= v0.4.0]
        leap_update : If True, when the leap second file is more than 6 months old will attempt to download a new one.
                      If False, then will just give a warning message. Default is True. [Not used with versions >= v0.4.0]

    OUTPUT:
        corr_time : BJDTDB time
        warning : Warning and Error message from the routine.
        status : Status regarding warning and error message. Returns the following -
                0 - No warning or error.
                1 - Warning message.
                2 - Error message.

    Example:
    >>> from astropy.time import Time
    >>> JDUTC = Time(2458000, format='jd', scale='utc')
    >>> utc_tdb.JDUTC_to_BJDTDB(JDUTC,hip_id=8102, lat=-30.169283, longi=-70.806789, alt=2241.9)
    (array([ 2458000.00505211]), [[], []], 0)

    '''

    corr_time = []
    warning = []
    error = []
    status = 0

    # Check for JDUTC type
    if type(JDUTC)!=Time:
         warning += [['Warning: Float JDUTC entered. Verify time scale (UTC) and format (JD)']]
         JDUTC=Time(JDUTC, format='jd', scale='utc')

    if JDUTC.isscalar:
        JDUTC = Time([JDUTC])

    star_par = {'ra':ra,'dec':dec,'pmra':pmra,'pmdec':pmdec,'px':px,'rv':rv,'epoch':epoch}
    star_simbad = {'ra':None,'dec':None,'pmra':None,'pmdec':None,'px':None,'rv':None,'epoch':None}
    star_hip = {}
    star_zero = {'ra':0.,'dec':0.,'pmra':0.,'pmdec':0.,'px':0.,'rv':0.,'epoch':2451545.0}
    star_output = {}


    if starname:
        star_simbad,warning1 = get_stellar_data(starname)
        warning += warning1
    if hip_id:
        if starname:
            warning += ['Warning: Querying SIMBAD and Hipparcos Catalogue']
        star_hip = find_hip(hip_id)


    star_output = star_simbad.copy()
    star_output.update({k:star_hip[k] for k in star_hip if star_hip[k] is not None})
    star_output.update({k:star_par[k] for k in star_par if star_par[k] is not None})
    star_output.update({k:star_zero[k] for k in star_zero if star_output[k] is None})
    warning+=['Following are the stellar positional parameters being used - ',star_output]


    if obsname:
        loc = EarthLocation.of_site(obsname)
        lat = loc.lat.value
        longi = loc.lon.value
        alt = loc.height.value
        warning += [['Warning: Taking observatory coordinates from Astropy Observatory database. Verify precision. Latitude = %f  Longitude = %f  Altitude = %f'%(lat,longi,alt)]]
    else:
        loc = EarthLocation.from_geodetic(longi, lat, height=alt)

    for jdutc in JDUTC:
        a = _JDUTC_to_BJDTDB(JDUTC=jdutc,
                 loc=loc,
                 ephemeris=ephemeris, **star_output)
        corr_time.append(a[0])
        warning.append(a[1])
        error.append(a[2])


    # Status messages to check for warning or error
    if not all(corr_time): error += ['Check inputs. Error in code']
    if any(error):   status |= 2
    if any(warning): status |= 1
    # Convert corrected from list to numpy array
    corr_time = np.array(corr_time)

    return corr_time, warning+error, status


def _JDUTC_to_BJDTDB(JDUTC,
    ra=0.0, dec=0.0, epoch=2451545.0, pmra=0.0, pmdec=0.0, px=0.0, rv=0.0,
    loc=None,
    ephemeris='de430', leap_dir=os.path.join(os.path.dirname(__file__),'data'), leap_update=True):

    '''
    Time conversion between JDUTC to BJDTDB. See Eastman et al. (2010)
    This code is precise to about 10 ms

    See JDUTC_to_BJDTDB() for parameter description.

    '''

    # Convert times to obtain TDB and TT
    JDTDB, JDTT, warning, error = JDUTC_to_JDTDB(JDUTC)
    clock_corr = (JDTDB.jd - JDUTC.jd) * 86400.

    ##### NUTATION, PRECESSION, ETC. #####

    r_pint, v_pint = PINT.gcrs_posvel_from_itrf(loc, JDUTC, JDTT)

    r_eci = r_pint[0]  # [m]
    v_eci = v_pint[0]  # [m/s]

    ##### EPHEMERIDES #####

    earth_geo = get_body_barycentric_posvel('earth', JDTDB, ephemeris=ephemeris) # [km]
    r_obs = r_eci + earth_geo[0].xyz.value*1000. # [m]
    v_geo = earth_geo[1].xyz.value*1000./86400.  # [m/s]

    # Relativistic Addition of Velocities
    v_obs = (v_eci+v_geo) / (1.+v_eci*v_geo/c**2) # [m/s]

    # calculate the Einstein delay relative to the geocenter
    # (TDB accounts for Einstein delay to geocenter)
    einstein_corr = np.sum(r_eci*v_geo)/(c*c)

    ##### Convert Star RA DEC to R0hat vector #####

    r0hat = np.array([math.cos(ra*np.pi/180.)*math.cos(dec*np.pi/180.),
                      math.sin(ra*np.pi/180.)*math.cos(dec*np.pi/180.),
                                              math.sin(dec*np.pi/180.)])
    # Eq 14 to 17 from Wright and Eastman 2014
    up = [0., 0., 1.]
    east = np.cross(up, r0hat)
    east = east / math.sqrt(sum(east*east))
    north = np.cross(r0hat, east)
    mu = (pmra*east+pmdec*north)/pctoau/1000 # Divided by 1000 since the Proper motion is in milli-arcseconds.


    ##### Stellar position corrected for motion #####

    epoch0 = 2000. + (epoch-2451545.)/365.25
    yearnow = 2000. + (JDTDB.jd-2451545.)/365.25

    T = yearnow - epoch0                           # [years]
    vpi = rv/1.e3 * kmstoauyr * (px/1.e3/pctoau)   # [rad/yr]
    vel = mu + vpi*r0hat                           # [rad/yr] (m in AA)
    r = r0hat + vel*T                              # [rad]    (p1 in AA)
    rhat = r / math.sqrt(sum(r*r))

    # Geometric Correction
    geo_corr = np.sum(r_obs*rhat)/c

    delta_t = geo_corr + clock_corr + einstein_corr
    result = JDUTC.jd+delta_t/86400.

    return result, warning, error


def JDUTC_to_JDTDB(utctime, leap_update=True, fpath=fpath):
    '''
    v0.4.0 onwards
    20210303
    Using Astropy for leap seconds instead of maintaining/updating a separate files.

    DEPRECATED
    Convert JDUTC to JDTDB (Barycentric Dynamical Time)
    INPUT:
        utctime : Enter UTC time as Astropy Time Object. In UTC Scale.
        fpath : Path to where the file would be saved. Default is script directory.
        leap_update : If True, when the leap second file is more than 6 months old it will attempt to download a new one.
                      If False, then will just give a warning message.  Default is True. [Not used with versions >= v0.4.0]

    OUTPUT:
        JDTDB : Julian Date Barycentric Dynamic Time (Astropy Time object)
        JDTT: Julian Date Terrestrial Dynamic time (Astropy Time object)
        warning,error : Warning and Error message if any from the routine
    '''
    warning = []
    error = []

    if astropy.__version__ < '4.0.4':
        error += [
            "ERROR: Using Astropy version < 4.0.4 might not have the latest leap second file. Please update to a newer Astropy version to ensure updated leap seconds. "]

    return utctime.tdb, utctime.tt, warning, error