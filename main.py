"""
    File name: main.py
    Author: Peter Steiglechner (https://orcid.org/0000-0002-1937-5983)
    Date created: 01 December 2020
    Date last modified: 04 May 2021
    Python Version: 3.8
"""

import sys
from time import time
from pathlib import Path
import importlib
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from agents import Agent
from create_map import Map
from saving import *


class Model:
    """
    An Agent-Based Model (ABM) that simulates the spatial and temporal dynamics of household agents on Easter Island
    and their interactions with the natural environment through resource consumption prior to European arrival.

    Short Summary
    =============
    The environment is encoded on a 2D discretised map with real geographic and orographic features. Agents are
    represented by households, which comprise a variable number of individuals. Households rely on two limited
    resources: (1) palm trees, considered here a primary, non-renewable resource for essential tools, firewood,
    building material, sugary sap, etc. and (2) cultivated sweet potatoes, which constituted an important source of
    carbohydrates and water on the island. Households use these resources by cutting trees and by creating gardens (
    i.e., cultivating cleared, arable land available in their immediate surrounding). The growth or decline of
    households depends on the success with which they can obtain these resources. Households adapt to the changing
    environment and to the growing population in three ways. First, a household splits into two when it becomes too
    large and one of the two relocates in a different place. Second, households relocate when resources become scarce
    in their current location. Their moving behaviour is determined by resource availability and certain features of
    the environment, including elevation and distance from the three major lakes (Rano Kau, Rano Raraku,
    and Rano Aroi). Third, in a response to the declining number of trees, households adapt their resource preference
    from a resource combination dominated by non-renewable trees to a combination dominated by stable cultivation of
    sweet potatoes. In summary, the interaction between agents and the natural environment and the adaptive response
    of agents, shape settlement patterns and population dynamics on the island.

    Time in the Model
    =================
    In accordance with suggestions by Bahn and Flenley (2017) the simulations start with two households (comprising a
    total population of 40 individuals) positioned in the proximity of Anakena Beach in the northern part of the
    island in the year 800 A.D., thus, mimicking the arrival of the first Polynesian settlers. Model updates occur
    asynchronously on time steps of one year until 1800 A.D.. The model does not include processes such as spreading
    of diseases or slavery that were introduced after the discovery of the island by European voyagers in the 18th
    century.

    References and Further Reading
    ==============================
    Bahn, P., & Flenley, J. (2017). Easter Island, Earth Island: The Enigmas of Rapa Nui. Rowman & Littlefield.
    Puleston CO, Ladefoged TN, Haoa S, Chadwick OA, Vitousek PM, Stevenson CM (2017) Rain, sun, soil, and sweat:
        A consideration of population limits on Rapa Nui (Easter Island) before European contact. Frontiers in
        Ecology and Evolution DOI 10.3389/fevo.2017.00069
    Rull, V. (2020), The deforestation of Easter Island. Biol Rev, 95: 124-141. https://doi.org/10.1111/brv.12556

    Functions
    =========
    - init : initiate the model by setting constants and parameters and creating the map,
    - run : run one simulation
    - init_agents : initialise the n_agents_arrival agents
    - observe : store agent's traits, state of the environment and aggregate variables for one time step
    - step : proceed one time step
    - P_cat : return the penalty following a logistic function of an evaluation variable for cells in a specified
        category.
    """
    def __init__(self, model_folder, model_seed, params_const, params_sensitivity, params_scenarios):
        """
        Initiate the model by seting constants and parameters and creating the map

        Parameters
        ----------
        folder : str
            folder for storing data
        seed: int
            seed value
        params_const : dict
            all constant parameters that are not changed in the sensitivity analysis
        params_sensitivity : dict
            parameters changed for different runs in the sensitivity analysis:
            initial tree pattern, arrival time and population growth rate, resource requirements,
        params_scenarios : dict
            parameters for the three scenarios: unconstrained, partlyconstrained, fullyconstrained
        """
        self.seed = model_seed  # Seed
        np.random.seed(self.seed)

        self.p_arrival = params_const["p_arrival"]  # population at arrival
        self.time_end = params_const["time_end"]  # end time of the simulation
        self.moving_radius_arrival = params_const["moving_radius_arrival"]    # Radius after arrival in Anakena Beach
        self.arability_well = params_const["arability_well"]    # arability index of well-suited cells
        self.arability_poor = params_const["arability_poor"]    # arability index of poorly suited cells
        self.garden_area_m2 = params_const["garden_area_m2"]    # size of a garden in m^2
        self.gridpoints_y = params_const["gridpoints_y"]
        self.gridpoints_x = params_const["gridpoints_x"]
        self.n_trees_arrival = params_const["n_trees_arrival"]
        self.t_pref_max = params_const["t_pref_max"]
        self.t_pref_min = params_const["t_pref_min"]
        self.p_splitting_agent = params_const["p_splitting_agent"]
        self.p_remove_threshold = params_const["p_remove_threshold"]
        self.s_equ = params_const["satisfaction_equ"]
        self.evaluation_thresholds = params_const["evaluation_thresholds"]
        self.alpha = params_const["alpha"]

        # Params for sensitivity analysis
        self.c_req_pp = params_sensitivity["c_req_pp"]  # max. cultivation requirement in Nr of 1000 m^2 gardens per P
        self.t_req_pp = params_sensitivity["t_req_pp"]  # max. tree requirement in trees per person per year
        self.time_arrival = params_sensitivity["time_arrival"]  # time of arrival in A.D.
        self.max_p_growth_rate = params_sensitivity["max_p_growth_rate"]
        self.map_tree_pattern_condition = params_sensitivity["map_tree_pattern_condition"]
        self.droughts_rano_raraku = params_sensitivity["droughts_rano_raraku"]  # Drought periods of Rano Raraku

        # Params for scenarios: Aggregate, Homogeneous, Constrained, Full
        self.p_split_threshold = params_scenarios["p_split_threshold"]
        self.n_agents_arrival = params_scenarios["n_agents_arrival"]  # nr of agents at arrival
        self.r_t = params_scenarios["r_t"]
        self.r_c = params_scenarios["r_c"]
        self.gamma = params_scenarios["gamma"]

        self.time_range = np.arange(self.time_arrival, self.time_end + 1)
        self.time = self.time_arrival

        # ==== Create the Map ====
        print("Creating the map")
        # Bounding Box of el image
        bbox_el_image = [-109.465, -27.205, -109.2227, -27.0437]
        # Shift the Bounding Box of Elevation Picture to the one of Puleston 2017
        delta_puleston_el_bbox = (0.013, 0.007, 0.003, -0.01)  # dlonmin, dlatmin, dlonmax, dlatmax
        puleston_bbox = [c + d for c, d in zip(bbox_el_image, delta_puleston_el_bbox)]

        self.map = Map(self, "Map/elevation_EI.tif", "Map/slope_EI.tif", "Map/puleston2017.jpg", bbox_el_image,
                       puleston_bbox)
        self.map.check_drought(self.time)

        # === Preparing for Storage ===
        self.folder = model_folder  # folder for saving
        self.schedule = []  # list of all agents.

        # Environmental variables
        self.trees = np.empty([len(self.time_range), self.map.n_triangles_map], dtype=np.int)
        self.gardens = np.empty([len(self.time_range), self.map.n_triangles_map], dtype=np.int)
        self.population = np.empty([len(self.time_range), self.map.n_triangles_map], dtype=np.int)
        self.clearance = np.empty([len(self.time_range), self.map.n_triangles_map], dtype=np.int)
        self.lakes = np.zeros([len(self.time_range), self.map.n_triangles_map], dtype=bool)

        # agent variables: for every agent at each time
        self.agents_stats_columns = ["time", "id", "x", "y", "pop", "n_gardens", "t_pref"]
        self.agents_stats = None

        # aggregate variables: one value for each timestep
        self.n_agents_arr = []
        self.resource_motivated_moves_arr = []
        self.excess_deaths_arr = []
        self.excess_births_arr = []
        self.mean_satisfactions_arr = []
        self.std_satisfactions_arr = []
        self.n_tree_unfilled_arr = []
        self.n_cult_unfilled_arr = []
        self.mean_tree_fill_arr = []
        self.mean_cult_fill_arr = []
        self.fractions_poor_vs_well_arr = []

        self.resource_motivated_moves = 0
        self.excess_deaths = 0
        self.excess_births = 0
        self.max_agent_index = 0

        self.time = np.nan
        return

    def run(self):
        """
        Run one simulation

        Steps
        =====
            - Initialise agents
            - Loop through each time step
                - Make one time step (update all agents sequentially)
                - check whether there is a drought of Rano Raraku
        """
        self.time = self.time_arrival
        self.init_agents()
        self.observe(self.time_arrival)
        for t in np.arange(self.time_arrival+1, self.time_end+1):
            self.time = t
            self.step()
            self.observe(t)
            self.map.check_drought(t)
        save_all(self)
        return

    def init_agents(self):
        """
        Initialise the n_agents_arrival agents

        The simulation starts with two agents (with a total population of 40 individuals) settling in proximity to
            Anakena Beach in the North part of the island in the year t_0 = 800 A.D., following [Bahn2017].
        We assume, they erect a settlement nearby within radius moving_radius_arrival
        """
        for i in range(self.n_agents_arrival):
            # Arrival point is at Anakena Beach
            x, y = self.map.midpoints_map[self.map.anakena_ind_map]
            # Create an agent, x,y are at Anakena Beach, population p is the initial population over the initial agents
            ag = Agent(self, x, y, int(self.p_arrival/self.n_agents_arrival))
            ag.cell = self.map.anakena_ind_map
            ag.cell_all = self.map.anakena_ind

            # For storage purposes:
            # Increase agent number in the cell of Anakena Beach by 1
            # self.map.agNr_c[ag.cell] += 1
            # Increase population size in the cell of Anakena Beach by the agent's population p
            self.map.pop_cell[ag.cell] += ag.p

            # increase the running agent indices (including dead agents)
            self.max_agent_index += 1

            # Move the agent to a new location, within the radius specified by moving_radius_arrival around anakena
            ag.move(self.map.circ_inds_anakena)

            # Update tree preference and consequently resource requirements
            ag.update_t_pref()
            ag.calc_resource_req()

            # Add agent to schedule (list of agents)
            self.schedule.append(ag)
        return

    def observe(self, t):
        """
        At the current time step, store agents' traits, state of the environment and aggregate variables for one
            time step
        """
        store_agents_values(self, t)  # agent's traits
        store_dynamic_map(self, t)  # state of the environment
        store_agg_values(self)  # aggregate variables

        # Print out:
        n_agents = len(self.schedule)
        tot_pop = np.sum(self.map.pop_cell)
        tot_trees = np.sum(self.map.trees_map)
        tot_gardens = np.sum(self.map.cultivated_gardens)
        print("t={}: n_ags={}; p={}; tr={}; f={}".format(t, n_agents, tot_pop, tot_trees, tot_gardens))
        return

    def step(self):
        """
        Proceed one time step

        Steps
        =====
            - sort agents randomly
            - perform updates of each agent sequentially
        """
        n_agents = len(self.schedule)
        if n_agents == 0:
            return
        else:
            # Update without  replacement
            agents_order = np.random.choice(self.schedule, n_agents, replace=False)
            for selected_agent in agents_order:
                selected_agent.update()
        return

    def P_cat(self, x, cat, infty_penalty=False, **kwargs):
        """
        Return the penalty following a logistic function of an evaluation variable for cells in a specified category.

        Idea
        ====
        -For each evaluation category (cat) an agent defines a categorical penalty, P_{cat}(c), and evaluates
            all cells accordingly.
        - The more unfavourable the conditions are in a cell, the higher the cell's penalties.
        - The penalties, P_{cat}(c), depend logistically on the correlated, underlying geographic condition
            ranging from 0 (very favourable condition) to 1 (very unfavourable).
        - The penalty is set to infinity to inhibit a move to those cells c, in which the agent can not fill either
            its current tree or cultivation requirement for at least the upcoming year if it would move to this cell.

        Parameters
        --------
        cat : string
            can be one of "ld", "pd", "or", "tr", "cu"
            if cat is "tr" or "cu", then kwargs needs to have "ag": kwargs["ag"] = object of class Agent
        x : array of floats
            values of the evaluation criteria for each cell
        infty_penalty : str, default: "none"
            "smaller than x99" if penalty is set to 1e10 for x<x99
            or "larger than x99"  if penalty is set to 1e10 for x>x99
            or "none" if penalty approaches 1 for x becoming "less favourable" than x99.
        **kwargs :
             "ag": instance of class Agent

        Returns
        -------
        penalties : array of floats
            the penalties for each cell on land for the category between 0 and 1 or 1e10

        """
        if cat == "tr" or cat == "cu":
            # calculate the tree or cultivation 99% evaluation threshold.
            x99 = self.evaluation_thresholds[cat + "99"](kwargs["ag"], self)
        else:
            # read the lake distance, orographic or population density 99% evaluation threshold
            x99 = self.evaluation_thresholds[cat + "99"]
        # read the 1% evaluation threshold
        x01 = self.evaluation_thresholds[cat + "01"]
        # steepness parameter of logistic function between the two thresholds
        k = 1 / (0.5 * (x99 - x01)) * np.log(0.99 / 0.01)
        # logistic function of the variable x in category in a cell with the specified thresholds.
        penalties = 1 / (1 + np.exp(-k * (x - 0.5 * (x99 + x01))))

        if infty_penalty == "smaller than x99":
            # e.g. if there are too few trees or arable land (i.e. x<x99) in a cell
            # penalty = infty if x < x99
            penalties *= np.ones_like(x) + 1000 * (x < x99)
        if infty_penalty == "larger than x99":
            # e.g. if there is too high elevation/slope (i.e. x>x99) in a cell
            # penalty = infty if x > x99
            penalties *= np.ones_like(x) + 1000 * (x > x99)
        return penalties


if __name__ == "__main__":
    """
    Example Usage
    =============
    '''
        python main.py default fullyconstrained 1
    '''
    - the first additional argument is the filename of parameter values for the different experiments tested in the 
        sensitivity analysis:
            params/sa/...  e.g. /params/sa/fullyconstrained
    - the second additional argument is the filename of parameter values for the specific scenario `unconstrained', 
        `partlyconstrained', or `fullyconstrained': 
            params/scenarios/... e.g. /params/scenarios/fullyconstrained
    - the third additional argument denotes the seed value used.
    
        
    """
    print("RUN: ", sys.argv)
    if len(sys.argv) < 4:
        print("Provide 3 arguments: python main.py default fullyconstrained 1")

    RUNSTART = time()

    sa_file = sys.argv[1]
    sa_mod = importlib.import_module("params.sa."+sa_file)
    print("sa.params_sensitivity: ", sa_mod.params_sensitivity)

    scenario_file = sys.argv[2]
    scenario_mod = importlib.import_module("params.scenarios." + scenario_file)
    print("scenarios.params_sensitivity: ", scenario_mod.params_scenario)
    # For the `Aggregate' Scenario, use params.scenario.aggregate and adjust the folder name!

    seed = sys.argv[3]

    const_file = "default_consts"  # "default_consts"
    consts_mod = importlib.import_module("params.consts."+const_file)
    print("const file", const_file)

    folder = "data/{}_{}_seed{}/".format(sa_file, scenario_file, str(seed))

    # === Save files ===
    # Save the current state of the implementation to the subfolder
    Path(folder).mkdir(parents=True, exist_ok=True)
    Path(folder + "/used_files/").mkdir(parents=True, exist_ok=True)
    for file in ["main.py", "create_map.py", "agents.py"]:
        shutil.copy(file, folder + "/used_files/")
    for file in ["consts/"+const_file, "sa/"+sa_file, "scenarios/"+scenario_file]:
        shutil.copy("params/"+file+".py", folder + "/used_files/")
    print("Working in ", folder, "; Python scripts and parameter files have been copied into subfolder used_files")

    m = Model(folder, int(seed), consts_mod.params_const, sa_mod.params_sensitivity, scenario_mod.params_scenario)
    m.run()

    RUNEND = time()
    print("Run time was: {} min".format((RUNEND-RUNSTART)/60))
