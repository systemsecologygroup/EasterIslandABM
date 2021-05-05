import numpy as np

params_const = {
    "p_arrival": 40,
    "time_end": 1900,
    "moving_radius_arrival": 1,
    "arability_well": 1,
    "arability_poor": 0.05,
    "garden_area_m2": 1000,
    "gridpoints_y": 50,
    "gridpoints_x": 75,
    "n_trees_arrival": 16e6,  # Mieth 2015
    "t_pref_max": 0.8,
    "t_pref_min": 0.2,
    "p_splitting_agent": int(12),
    "p_remove_threshold": int(6),
    "satisfaction_equ": 0.68844221,  # Puleston et al 2017
    "evaluation_thresholds":   # moving_evaluation_thresholds
        {
            "ld01": 0.5 ** 2 / (np.pi * 0.170 ** 2),  # Water: 1% penalty    (0.5km)^2 / (pi * rad_raraku^2)
            "ld99": 5.0 ** 2 / (np.pi * 0.170 ** 2),  # Water: 99% penalty   (5km)^2 / (pi * rad_raraku^2)
            "el01": 0,  # Elevation: 1% penalty     [m] prefer low elevation
            "el99": 300,  # Elevation: 99% penalty     [m] avoid 300-500m elevation
            "sl01": 0,  # Slope: 1% penalty     [degree] prefer no slope
            "sl99": 7.5,  # Slope: 99% penalty     [degree] avoid high slopes
            "pd01": 0,  # Population density: 1% penalty     [ppl/km^2] prefer no neighbours
            "pd99": 300,
            # Population density: 99% penalty    [ppl/km^2] Kirch (2010), Puleston et al. (2017), rough estimate of local population density in Hawaii and Maui
            "tr01": 1e5,  # Tree: 1% penalty     prefer if there are many trees around
            "tr99": lambda ag, m: ag.t_pref * m.t_req_pp * ag.p * m.s_equ,  # Tree: 99% penalty
            "cu01": 200,  # Farming: 1% penalty    prefer if there are many farming grounds around
            "cu99": lambda ag, m: (1 - ag.t_pref) * m.c_req_pp * ag.p * m.s_equ,  # Farming: 99% penalty
        },
    "alpha":
        {
            "ld": 0.2,
            "or": 0.2,
            "pd": 0.2,
            "tr": 0.2,
            "cu": 0.2
        },
}
