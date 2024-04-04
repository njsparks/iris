# %%
import numpy as np
from scipy import stats
from scipy import interpolate
import xarray as xr
import geopandas as gpd
from shapely.geometry import LineString, Point

# single argument wrapper for run. Useful for multiprocessing pool.map
def run_1arg(arg):
    print(arg["result"])
    out = run(arg["inputDat"], arg["result"], arg["seed"], arg['nY'])
    return out

def run(inputDict, results_path, seed, nY):
    # inputDict: parent tracks and parameters.
    # results_path: path where simulated tracks are saved.
    # seed: seed for RNG, for reproducibility.
    # nYears: Number of years to simulate.

    # seed RNG
    np.random.seed(seed)

    # number the input tcs for parent tracing
    i=0
    for tcy in inputDict["tcs"]:
        for tc in tcy:
            tc["n"] = i
            i=+1

    # run for nY years
    tcs_out_years = []
    for y in range(nY):
        
        # random selection of input tracks
        # flatten tc years for random tc selection
        parents = []
        for tcy in inputDict["tcs"]:
            parents.extend(tcy)
        count = gen_count(inputDict["count_parms"])
        tcs_in = np.random.choice(parents, count)

        # # select parent tracks serially, year for year. nY < nYobs 
        # tcs_in=inputDict["tcs"][y]

        tcs_out = []
        for tc in tcs_in:
            # generate new track from each parent
            tc_out = gen_perturbed_tc(
                tc,
                inputDict["kappa_parms_alg"],
                inputDict["kappa_parms_land_alg"],
                inputDict["lmimpi_parms"],
                inputDict["size_parms"],
                inputDict["pres_parms"],
            )
            tcs_out.append(tc_out)
        tcs_out_years.append(tcs_out)

    outData = {"tcs": tcs_out_years, "seed": seed}
    np.save(results_path, outData)
    return outData


def gen_perturbed_tc(tc, kappa_parms, kappa_land_parms, lmimpi_parms, size_parms,pres_parms):
    # Perturb one parent track (tc) using supplied model params. 
    
    # post-lmi (inclusive) parent
    lon_in = np.array(tc["lon"])[tc["lmix"]:]
    lat_in = np.array(tc["lat"])[tc["lmix"]:]
    # pre-lmi parent
    lon_in_pre = np.array(tc["lon"])[: tc["lmix"]]
    lat_in_pre = np.array(tc["lat"])[: tc["lmix"]]

    
    # Forward Track Model
    month = tc["month"]
    lon, lat = trackModel(lon_in, lat_in, month, lon_in_pre, lat_in_pre)
    
    # LMI Model
    mpi = get_mpi(month, lon[0], lat[0])["vmax"]
    mpi = max(33, mpi)

    # catch LMI below minVmax and set to minVmax
    minVmax = 33
    lmimpi = gen_lmimpi_uniform_noPred(lmimpi_parms)
    vmax0sim = mpi * lmimpi
    vmax0sim=max(minVmax,vmax0sim)
        
    ####################### post LMI model#######################

    nt = lon.size  # number of timesteps

    # used when TC is equatorward of 5 degs or moves to low MPI region
    expDecayFrac = np.exp(-3 / 6)

    # algebraic decay constant kappa (from S. Wang 2022 eq.4, inverse length units)
    kappa_alg = gen_kappa(kappa_parms, vmax0sim)
    # kappa_log = gen_kappa(kappa_parms_log,vmax0sim)
    # kappa_alg = 0
    kappa_alg_land = None
    waterfraction = None
    landfalls = []
    nLandfall = 0
    mpiDecay = False

    # vmax cutoff thresh = 17.5 m/s, decay model not valid below TS
    vmax_min = 17.5

    # init time series
    vmax = [vmax0sim] # begin decay at LMI model value
    mpis = [mpi]
    
    # isLand = [isPointLand(lon[0], lat[0])]
    isLand = [point_landfall({'lon':[lon[0]],'lat':[lat[0]]})]

    for i in range(nt - 1):
        mpis.append(get_mpi(month, lon[i], lat[i])["vmax"])
        # isLand.append(isPointLand(lon[i], lat[i]))
        isLand.append(point_landfall({'lon':[lon[i]],'lat':[lat[i]]}))
        
        # trigger fast mpi decay once mpi falls below threshold
        if (mpis[i] < 18) & np.isfinite(mpis[i]):
            mpiDecay = True

        # perform fast decay in equatorial region or when low mpi is triggered
        if (np.abs(lat[i]) < 5) or mpiDecay:
            vmax.append(vmax[i] * expDecayFrac)

        elif isLand[i]:
            # land decay (needs kappa_alg_land)
            if kappa_alg_land == None:  # first land timestep, new landfall
                nLandfall = nLandfall + 1
                waterfraction = get_water_fraction_3deg(lon[i], lat[i])
                kappa_alg_land = gen_kappa(kappa_land_parms, waterfraction)

                landfalls.append(
                    {
                        "n": nLandfall,
                        "t": i,
                        "lon": np.round(lon[i], 3),
                        "lat": np.round(lat[i], 3),
                        "waterfraction": np.round(waterfraction, 3),
                        "kappa": np.round(kappa_alg_land, 3),
                        "vmax0": np.round(vmax[-1], 3),
                    }
                )
            # calc decay
            v0 = vmax[-1]  # v0 changes every step, always previous step
            v3 = v0 / (1 + (v0 * 3 * kappa_alg_land))  # three hours later...
            vmax.append(v3)
        
        else:  # regular ocean decay
            
            # reset landfall params
            kappa_alg_land = None
            waterfraction = None
            
            v0 = vmax[-1]
            
            # algebraic decay (needs kappa_alg)
            vnew = v0 / (1 + (v0 * 3 * kappa_alg))  # three hours later...  
            vmax.append(vnew)

        # continue decay until either end of track is reached or vmax fall below thresh
        if vmax[-1] < vmax_min:
            break

    vmax = np.array(vmax)
    isLand = np.array(isLand)
    mpis = np.array(mpis)

    # ensure min(vmax) > vmax_min
    vx = np.where(vmax > vmax_min)[0][-1] + 1
    lon = lon[:vx]
    lat = lat[:vx]
    vmax = vmax[:vx]
    isLand = isLand[:vx]
    mpis = mpis[:vx]

    # size model
    rmw, r18, _ = gen_size_along_track(vmax, size_parms)

    # Forward Pressure model
    pmin = gen_Pmin(vmax, r18, lat, lon, month,pres_parms)

    # build full track from only forward track (i.e. no backtrack)
    lon_full=lon
    lat_full=lat
    isLand_full=isLand
    vmax_full=vmax
    rmw_full=rmw
    r18_full=r18
    pmin_full=pmin

    #Forward Landfall Flag
    flf_full=forward_landfall({'lon':lon_full,'lat':lat_full})

    # Build Output - full track
    tc_out = {
        "lon": np.round(lon_full, 4),
        "lat": np.round(lat_full, 4),
        "isLand": isLand_full,
        "vmax": np.round(vmax_full, 2),
        "pmin": np.round(pmin_full, 2),
        "mpi_0": np.round(mpi, 2),
        "month": month,
        "landfalls": landfalls,
        "n_parent": tc["n"],
        "mpiDecay": mpiDecay,
        "rmw": np.round(rmw_full, 1),
        "r18": np.round(r18_full, 1),
        "flf": flf_full,
    }

    return tc_out


# %% data lookups
data_dir = "../rundata/"
# Steering wind climatology lookup
uv500_mean = xr.load_dataset(data_dir + "era5_uv500_mm_clim_mean_1deg.nc")
def get_uv500clim(lon, lat, month):
    # return clim uv in units of deg per 3 hr
    lon = lon % 360
    lat = (lat + 90) % 180 - 90
    uv500n = uv500_mean.sel(month=month, longitude=lon,
                            latitude=lat, method="nearest")
    u = np.array(uv500n.u.values)
    v = np.array(uv500n.v.values)
    udegp3h = u / (111000 * np.cos(np.array(lat) * 3.314 / 180)) * 3600 * 3
    vdegp3h = v / 110000 * 3600 * 3
    return udegp3h, vdegp3h

# lookup water fraction in 3 degree radius
waterfrac_dat3 = xr.load_dataset(data_dir + "waterfraction_rad3deg_res1deg.nc")
def get_water_fraction_3deg(lon, lat):
    lon = (lon + 180) % 360 - 180
    lat = (lat + 90) % 180 - 90
    wf = waterfrac_dat3.interp(
        longitude=lon,
        latitude=lat,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    ).waterfraction.values
    return wf.item()

# surface pressure monthly climatology lookup
msl_mean = xr.load_dataset(data_dir + "era5_msl_mm_clim_mean.nc")
def get_Pclim(lons, lats, month):
    Pclims = []
    for lon, lat in zip(lons, lats):
        Pclims.append(
            msl_mean.sel(
                month=month, longitude=lon, latitude=lat, method="nearest"
            ).msl.values
        )
    return np.array(Pclims)


# MPI lookup
da_mm_clim = xr.open_dataset(
    data_dir + "mpi_vmax_mm_clim_halfdeg_2int.nc").vmax
def get_mpi(month, lon, lat):
    lon = lon % 360
    # mon mean mpi clim vmax
    vmax = da_mm_clim.sel(
        month=month, longitude=lon, latitude=lat, method="nearest"
    ).item()
    return {"vmax": vmax}

# %% sub models
# count model
def gen_count(countparms):
    # gen basin season count from poission dist using mean basin count
    lam = countparms
    count = np.random.poisson(lam)
    return count

# decay parameter model
def gen_kappa(f, x):
    # f is dict containing parameters for
    # y~mx+c+sige model, where y is log kappa and x is vmax0 or waterfraction
    logkappa = f["m"] * x + f["c"] + f["sige"] * np.random.randn()
    kappa = np.exp(logkappa)
    return kappa


# LMI model
def gen_lmimpi_uniform_noPred(qf):
    # qf is dict containing parameters for
    # y~lower+uniform*(upper-lower)
    lb = qf["lower"]
    ub = qf["upper"]
    lmimpi = lb + np.random.random() * (ub - lb)
    return lmimpi


# Pressure model
def gen_Pmin(Vmax, R18, lat, lon, month, pres_parms):
    Vmax = np.array(Vmax)
    R18 = np.array(R18)
    lat = np.array(lat)
    lon = np.array(lon)

    lon = lon % 360
    lat = (lat + 90) % 180 - 90

    f = np.abs(np.sin(lat * (3.142 / 180)))

    Pdef = (
        pres_parms["xconst"]
        + pres_parms["xVmax"] * Vmax
        + pres_parms["xVmax2"] * Vmax**2
        + pres_parms["xR18"] * R18
        + pres_parms["xf"] * f
    )

    sig = np.random.randn() * pres_parms['RMSE']
    Pclim = get_Pclim(lon, lat, month)
    Pmin = Pclim - Pdef + sig
    return Pmin


# Track model
def trackModel(lon, lat, month, lon_pre, lat_pre,back=False):
    # track pert parameters
    sig0 = 1.0  # std of inital position perturbation
    sigDelta = 0.075  # std of deg per 3 hr perturbation

    nExtrap = 10 * 8  # number of timesteps to extrapolate track (10 days)

    # Track
    dlon = lon - lon[0]
    dlat = lat - lat[0]

    # starting perturbation
    initPertAttempts=0
    initPertAttemptsMax=10
    while initPertAttempts<initPertAttemptsMax:

        lon0d = np.random.randn() * sig0
        lat0d = np.random.randn() * sig0

        lon0 = lon[0] + lon0d
        lat0 = lat[0] + lat0d

        if not point_landfall({'lon':[lon0],'lat':[lat0]}):
            break
        initPertAttempts+=1

    # cumulative track perturbation
    pertlon = np.cumsum(np.tile(np.random.randn(), lon.size)) * sigDelta
    pertlat = np.cumsum(np.tile(np.random.randn(), lat.size)) * sigDelta

    lon = lon0 + dlon + pertlon
    lat = lat0 + dlat + pertlat

    # extend track (using last 12 hr mean translation)
    # xtrap len, 4 (12 hrs) or len of series if shorter
    # combine perturbed postLMI track with preLMI track for calculating extrapolation vector
    lon_cat = np.concatenate((lon_pre, lon))
    lat_cat = np.concatenate((lat_pre, lat))
    xlen = min(4, len(lon_cat) - 1)
    dlon_ex = (lon_cat[-1] - lon_cat[-1 - xlen]) / xlen
    dlat_ex = (lat_cat[-1] - lat_cat[-1 - xlen]) / xlen

    for _ in range(nExtrap): 
        lon = np.append(lon, lon[-1] + dlon_ex)
        lat = np.append(lat, lat[-1] + dlat_ex)

    lon, lat = relaxTrack2clim(lon, lat, month, nExtrap,back=back)

    return lon, lat

def relaxTrack2clim(lon, lat, month, n,back):
    # relax post parent translation speed to monthly clim uv500 over final n tsteps
    lonex = np.array(lon[-(n + 1):])  # extrapolated component
    latex = np.array(lat[-(n + 1):])
    dlonex = np.diff(lonex)[-1]
    dlatex = np.diff(latex)[-1]
    lonb = lon[:-n]  # base component
    latb = lat[:-n]

    # trigonometric transistion to clim winds by m timesteps
    m = 1.0 * n
    climWeight = np.concatenate(
        (
            np.sin(3.1415 / 2 * (np.linspace(0, 1, int(np.floor(m))))),
            np.ones(int(n - np.ceil(m))),
        )
    )

    for i in range(n):
        dlonc, dlatc = get_uv500clim(lonb[-1], latb[-1], month)
        # reverse for back trajectories
        if back:
            dlonc=-dlonc
            dlatc=-dlatc
        lonb = np.append(
            lonb, lonb[-1] + dlonex *
            (1 - climWeight[i]) + dlonc * climWeight[i]
        )
        latb = np.append(
            latb, latb[-1] + dlatex *
            (1 - climWeight[i]) + dlatc * climWeight[i]
        )
    return lonb, latb

# size model
def sim_mvn(fit, N=1):
    # bivariate normal distribution model gen
    out = np.atleast_2d(
        stats.multivariate_normal.rvs(
            mean=[fit["m1"], fit["m2"]], cov=fit["c"], size=N)
    )
    return out[:, 0], out[:, 1]

def sim_mlr(fit, x1, x2):
    # multiple linear regression model gen
    y = (
        x1 * fit["m1"]
        + x2 * fit["m2"]
        + fit["c"]
        + fit["sige"] * np.random.randn(len(x1))
    )
    return y

def calc_a(vmax, r18, rmw):
    # calc modified Rankine shape parameter
    a = np.log(vmax / 17.5) / np.log(r18 / rmw)
    return a

def gen_sizes(vmax_init, fits):
    # generate initial (LMI) and final (vmax=25) size params
    vmax_init = np.atleast_1d(vmax_init)
    logrmw, logr18 = sim_mvn(fits["fin"], len(vmax_init))
    rmw_fin = np.exp(logrmw)
    rmw_fin = np.clip(rmw_fin, 5, 300)
    r18_fin = np.exp(logr18)
    r18_fin = np.clip(r18_fin, rmw_fin + 1, 1200)

    a_fin = calc_a(25, r18_fin, rmw_fin)
    a_fin = np.clip(a_fin, 0.1, 1.5)

    rmw_init = sim_mlr(fits["rmw_init"], rmw_fin, vmax_init)
    rmw_init = np.clip(rmw_init, 5, 300)

    a_init = sim_mlr(fits["a_init"], a_fin, rmw_init)
    a_init = np.clip(a_init, 0.25, 2)
    r18_init = rmw_init * np.exp(np.log(vmax_init / 17.5) / a_init)
    r18_init = np.clip(r18_init, rmw_init, 600)
    r18_init = np.clip(r18_init, rmw_init, rmw_init * 40)

    return (rmw_init, rmw_fin, r18_init, r18_fin, a_init, a_fin)


def gen_size_along_track(vmax, sizefits):
    # interpolate and extrapolate params along track
    rmw_init, rmw_fin, r18_init, r18_fin, a_init, a_fin = gen_sizes(
        vmax[0], sizefits)
    x25 = np.argmin(np.abs(vmax - 25))
    if x25 > 0:
        f = interpolate.interp1d(
            [0, x25], [rmw_init[0], rmw_fin[0]], fill_value="extrapolate"
        )
        rmw = f(np.arange(0, len(vmax)))
        f = interpolate.interp1d(
            [0, x25], [r18_init[0], r18_fin[0]], fill_value="extrapolate"
        )
        r18 = f(np.arange(0, len(vmax)))
    else:
        rmw = np.ones(len(vmax)) * rmw_fin[0]
        r18 = np.ones(len(vmax)) * r18_fin[0]

    a25 = calc_a(vmax[x25], r18_fin, rmw_fin)
    # for values below vmax=25 (after size mod 'final state'), determine r18 from rmw and a(vmax=25)
    rmw[rmw < 5] = 5
    v25x = vmax < 25
    rmw[v25x] = rmw[x25]
    r18[v25x] = rmw[v25x] / np.exp(np.log(17.5 / vmax[v25x]) / a_fin)
    return rmw, r18, a25

# load landmask - gshhs low-res modified approximate that used by IBTrACS
landmask_shp=gpd.read_file('../rundata/gshhs_l_ibtracs')
def forward_landfall(tc):
    tclines=[]
    lon=np.array(tc['lon'])
    lat=np.array(tc['lat'])

    lon = (lon + 180) % 360 - 180
    lat = (lat + 90) % 180 - 90

    n=len(lon)
    # if n>1:
    for i in range(n-1):
        tclines.append(LineString([[lon[i], lat[i]], [lon[i+1], lat[i+1]]]))
    tclines=np.array(tclines,dtype=object)
    lfs = [landmask_shp.intersects(tcline).any() for tcline in tclines]
    lfs.append(landmask_shp.contains(Point(lon[-1],lat[-1])).any())
    lfs=np.array(lfs)
    return lfs

def point_landfall(tc):
    tcpoints=[]
    lon=np.array(tc['lon'])
    lat=np.array(tc['lat'])
    lon = (lon + 180) % 360 - 180
    lat = (lat + 90) % 180 - 90
    for i in range(len(lon)):
        tcpoints.append(Point([lon[i], lat[i]]))
    lfs=np.array([landmask_shp.contains(tcpoint).any() for tcpoint in tcpoints])
    return lfs
