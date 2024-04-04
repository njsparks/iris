# %%
import numpy as np

# define model parameters

# count parameters by basin
basin_counts_parms = {
    'NA': 6.738095238095247,
    'WP': 15.785714285714276,
    'EP': 9.714285714285715,
    'NI': 1.6428571428751202,
    'SI': 8.285714285714283,
    'SP': 5.095238095238105,
}

# relative intensity parameters
lmimpi_parms = {
    "upper": 1.0,
    "lower": 0.4,
}

# # decay model parameters
kappa_parms_alg = {
    'm': -0.026513348208160512,
    'c': -6.93647926110273,
    'sige': 0.7633330790088085,
}
# land decay model parameters
kappa_parms_land_alg = {
    'm': -2.355440240663073,
    'c': -5.46010613014801,
    'sige': 0.5809757413838003,
}

# size model parameters
size_parms = {
    'fin': {'m1': 3.815611,
            'm2': 5.035934,
            'c': ([[0.20426969, 0.11513165],
                   [0.11513165, 0.19507251]]),
            },
    'rmw_init': {'c': 38.824883,
                 'm1': 0.18688466,
                 'm2': -0.34017152,
                 'sige': 13.492027},
    'a_init': {'c': 0.42507052,
               'm1': 0.11444488,
               'm2': 0.0036998002,
               'sige': 0.15790361},
}

# pressure model parameters
pres_parms = {
    "RMSE": 6.9210536717031745,
    "xconst": -16.926547,
    "xVmax": 0.932628,
    "xVmax2": 0.006549,
    "xR18": 0.029484,
    "xf": 12.467399,
}

# Parent tracks
parent_tracks = np.load("../rundata/parent_tracks_IBTrACS.npy", allow_pickle=True).item()

basins = [
    "NA",
    "WP",
    "EP",
    "NI",
    "SI",
    "SP",
]

# build IRIS inputs by basin
inputs = {}
for basin in basins:
    inputDat = {
        "basin": basin,
        "tcs": parent_tracks[basin],
        "count_parms": basin_counts_parms[basin],
        "kappa_parms_alg": kappa_parms_alg,
        "kappa_parms_land_alg": kappa_parms_land_alg,
        "lmimpi_parms": lmimpi_parms,
        "size_parms": size_parms,
        "pres_parms": pres_parms,
    }
    inputs[basin] = inputDat
# %%
