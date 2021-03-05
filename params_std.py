import numpy as np





params_const = {
    "seed": 1,
    "n_agents_arrival": 2,
    "p_arrival": 40,
    "time_end": 1900,
    "moving_radius_arrival": 1,
    "droughts_rano_raraku": [[800, 1200], [1570, 1720]],
    "f_pi_well": 1,
    "f_pi_poor": 0.05,
    "garden_area_m2": 1000,
    "gridpoints_y": 50,
    "gridpoints_x": 75,
    "n_trees_arrival": 16e6,  # Mieth 2015
    "t_pref_max": 0.8,
    "t_pref_min": 0.2,
    "p_splitting_agent": int(12),
    "p_split_threshold": int(36),
    "p_remove_threshold": int(6),
    "satisfaction_equ": 0.68844221,  # Puleston et al 2017
    "evaluation_thresholds":   # moving_evaluation_thresholds
        {
            "w01": 0.5 ** 2 / (np.pi * 0.170 ** 2),  # Water: 1% penalty    (0.5km)^2 / (pi * rad_raraku^2)
            "w99": 5.0 ** 2 / (np.pi * 0.170 ** 2),  # Water: 99% penalty   (5km)^2 / (pi * rad_raraku^2)
            "el01": 0,  # Elevation: 1% penalty     [m] prefer low elevation
            "el99": 300,  # Elevation: 99% penalty     [m] avoid 300-500m elevation
            "sl01": 0,  # Slope: 1% penalty     [degree] prefer no slope
            "sl99": 7.5,  # Slope: 99% penalty     [degree] avoid high slopes
            "pd01": 0,  # Population density: 1% penalty     [ppl/km^2] prefer no neighbours
            "pd99": 300,
            # Population density: 99% penalty    [ppl/km^2] Kirch (2010), Puleston et al. (2017), rough estimate of local population density in Hawaii and Maui
            "tr01": 0,  # Tree: 1% penalty     [m] prefer low elevation
            "tr99": lambda ag, m: ag.t_pref * m.t_req_pp * ag.p * m.s_equ,  # Tree: 99% penalty
            "f01": 0,  # Farming: 1% penalty
            "f99": lambda ag, m: (1 - ag.t_pref) * m.f_req_pp * ag.p * m.s_equ,  # Farming: 99% penalty
        },
    "alpha":
        {
            "w": 0.2,
            "g": 0.2,
            "pd": 0.2,
            "tr": 0.2,
            "f": 0.2
        },
}

n_fix = "low"
f_req_pp = 6.79 if n_fix == "low" else 1.95  # TODO
# "low": 1/[(1.46 Tons/ha/year) * 0.1 (ha/gardens) * (2809 kcal/day / (Tons/year)  / (2785 kcal/day/person)]
#       1/[1.46 * 0.1 * 2809 * 2785] * [tons * ha * kcal * year * day * person / (ha year gardens day year kcal)]
#       1/[1.46 * 0.1 * 2809 * 2785] * [1/ (tons * ha * kcal * year * day * person / (ha year gardens day Tons kcal))]
#       6.79 [1/ * (person / gardens)]
#       6.79 gardens/person
# "high": 1/(5.09 * 0.1 * 2809/2785) [gardens/person]
#       1.95 gardens/person

params_sensitivity = {
    "t_req_pp": 10,
    "f_req_pp": f_req_pp,
    "time_arrival": 800,
    "max_p_growth_rate": 1.007,  # Bahn and Flenley 2017

    "map_tree_pattern_condition":
        {
            #"name": "Uniform",  # "Mosaic"
            "max_el": 450,
            "max_sl": 10,
            "tree_decrease_lake_distance": 0  # "Uniform"=0, "Mosaic"=1
        }
}

params_scenario = {
    "r_t": 2,
    "r_f": 1,
    "gamma": 20,  # 0
}
