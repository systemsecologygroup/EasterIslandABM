"""

"""
import sys

from create_map import Map
from agents import Agent
from saving import *

# == CONSTANTS ==


# == MODEL ==




class Model():
    def __init__(self, folder, seed, params_const, params_sensitivity, params_scenarios, **kwargs):
        """

        """
        self.seed = seed  # Seed
        np.random.seed(self.seed)

        self.n_agents_arrival = params_const["n_agents_arrival"]  # nr of agents at arrival
        self.p_arrival = params_const["p_arrival"]  # population at arrival
        self.time_end = params_const["time_end"]  # end time of the simulation
        self.moving_radius_arrival = params_const["moving_radius_arrival"]    # Radius after arrival in Anakena Beach
        self.droughts_rano_raraku = params_const["droughts_rano_raraku"]  # Droughts of Rano Raraku: list of droughts with tuple of start and end date
        self.f_pi_well = params_const["f_pi_well"]    # farming productivity index of well-suited cells
        self.f_pi_poor = params_const["f_pi_poor"]    # farming productivity index of poorly suited cells
        self.garden_area_m2 = params_const["garden_area_m2"]    # size of a garden in m^2
        self.gridpoints_y = params_const["gridpoints_y"]
        self.gridpoints_x =  params_const["gridpoints_x"]
        self.n_trees_arrival = params_const["n_trees_arrival"]
        self.t_pref_max = params_const["t_pref_max"]
        self.t_pref_min = params_const["t_pref_min"]
        self.p_splitting_agent = params_const["p_splitting_agent"]
        self.p_split_threshold = params_const["p_split_threshold"]
        self.p_remove_threshold = params_const["p_remove_threshold"]
        self.s_equ = params_const["satisfaction_equ"]
        self.evaluation_thresholds = params_const["evaluation_thresholds"]
        self.alpha = params_const["alpha"]



        self.f_req_pp = params_sensitivity["f_req_pp"]   # max. farming requirement in Nr of 1000 m^2 gardens per person
        self.t_req_pp = params_sensitivity["t_req_pp"]  # max. tree requirement in trees per person per year

        self.time_arrival = params_sensitivity["time_arrival"]  # time of arrival in A.D.
        self.max_p_growth_rate = params_sensitivity["max_p_growth_rate"]

        self.map_tree_pattern_condition = params_sensitivity["map_tree_pattern_condition"]

        self.r_t = params_scenarios["r_t"]
        self.r_f = params_scenarios["r_f"]
        self.gamma = params_scenarios["gamma"]

        # ==== Create Map ====
        print("Creating the map")
        self.map = Map(self)
        self.map.check_drought(self.time_arrival)

        self.folder = folder  # "" if "folder" not in kwargs.keys() else kwargs["folder"]
        self.time_range = np.arange(self.time_arrival, self.time_end+1)

        self.schedule = []
        self.time = self.time_arrival
        self.resource_motivated_moves = 0

        # === Storage ===
        self.excess_deaths = 0
        self.excess_births = 0
        self.max_agent_index = 0

        #self.const_map_values = None
        self.trees = np.empty([len(self.time_range), self.map.n_triangles_map], dtype=np.int)
        self.gardens = np.empty([len(self.time_range), self.map.n_triangles_map], dtype=np.int)
        self.population = np.empty([len(self.time_range), self.map.n_triangles_map], dtype=np.int)
        self.clearance = np.empty([len(self.time_range), self.map.n_triangles_map], dtype=np.int)
        self.lakes = np.zeros([len(self.time_range), self.map.n_triangles_map], dtype=bool)

        self.agents_stats_columns = ["time", "id", "x", "y", "pop", "n_gardens", "t_pref"]
        self.agents_stats = None

        self.n_agents_arr = []
        self.resource_motivated_moves_arr = []
        self.excess_deaths_arr = []
        self.excess_births_arr = []
        self.mean_satisfactions_arr = []
        self.std_satisfactions_arr = []
        self.n_tree_unfilled_arr = []
        self.n_farm_unfilled_arr = []
        self.mean_tree_fill_arr = []
        self.mean_farm_fill_arr = []
        self.fractions_poor_vs_well_arr = []

        return

    def run(self):
        self.init_agents()
        self.observe(self.time_arrival)
        for t in np.arange(self.time_arrival+1, self.time_end+1):
            self.step(t)
            self.observe(t)
            self.map.check_drought(t)
        save_all(self)
        return


    def init_agents(self):
        """
        Initalise the n_agents_arrival agents.
        Agents arrive at Anakena Beach in the year specified by time_arrival.
        They erect a settlement nearby within radius moving_radius_arrival
        """
        for i in range(self.n_agents_arrival):
            # Arrival point is at Anakena Beach
            x, y = self.map.midpoints_map[self.map.anakena_ind_map]
            # Create an agent, x,y are at Anakena Beach, population p = initial population distributed on all initial agents
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
        store_agents_values(self, t)
        store_dynamic_map(self, t)
        store_agg_values(self)

        n_agents = len(self.schedule)
        tot_pop = np.sum(self.map.pop_cell)
        tot_trees = np.sum(self.map.trees_map)
        tot_gardens = np.sum(self.map.occupied_gardens)
        print("t={}: n_ags={}; p={}; tr={}; f={}".format(t, n_agents, tot_pop, tot_trees, tot_gardens))
        return


    def step(self, t):
        """
        update one time step

        i.e. perform update() for each agent in random order times
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

    def mu_mean(self, s):
        """ 
        return the mean growth rate given an agent's past-dependent satisfaction
        
        
        """
        if s >= self.s_equ:
            m_grow = (self.max_p_growth_rate - 1) / (1 - self.s_equ)
            return m_grow * (s - self.s_equ) + 1
        else:
            m_decl = 1/self.s_equ
            return m_decl * s


    def P_cat(self, x, cat, infty_penalty=False, **kwargs):
        """
        return the evaluation function for cells in a specified category.

        Parameters
        --------
        cat : string
            can be one of "w", "pd", "g", "tr", "f"
            if cat is "tr" or "f", then kwargs needs to have "ag": kwargs["ag"] = agent
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
            the penalities for each cell on land for the category between 0 and 1 or 1e10

        """
        if cat == "tr" or cat == "f":
            # calculate the tree or farming 99% evaluation threshold.
            x99 = self.evaluation_thresholds[cat+"99"](kwargs["ag"], self)
        else:
            # read the water, geography or population density 99% evaluation threshold
            x99 = self.evaluation_thresholds[cat+"99"]
        # read the 1% evaluation threshold
        x01 = self.evaluation_thresholds[cat+"01"]
        # steepness parameter of logistic function between the two thresholds
        k = 1/(0.5*(x99-x01)) * np.log(0.99/0.01)
        # logistic function of the variable x in category in a cell with the specified thresholds.
        penalties = 1/(1+np.exp(-k*(x-0.5*(x99+x01))))

        if infty_penalty =="smaller than x99":
            # e.g. if there are too few trees or arable land (i.e. x<x99) in a cell
            # penalty = infty if x < x99
            penalties *= np.ones_like(x) + 1000 * (x < x99)
        if infty_penalty == "larger than x99":
            # e.g. if there is too high elevation/slope (i.e. x>x99) in a cell
            # penalty = infty if x > x99
            penalties *= np.ones_like(x) + 1000 * (x > x99)
        return penalties


from time import time
from pathlib import Path
import importlib
import shutil
if __name__=="__main__":
    print("RUN: ", sys.argv)
    if len(sys.argv) < 4:
        print("Provide 3 arguments: python main.py default full 1")

    RUNSTART = time()

    sa_file = sys.argv[1]
    sa_mod = importlib.import_module("params.sa."+sa_file)
    print("sa.params_sensitivity: ", sa_mod.params_sensitivity)

    scenario_file = sys.argv[2]
    scenario_mod = importlib.import_module("params.scenarios." + scenario_file)
    print("scenarios.params_sensitivity: ", scenario_mod.params_scenario)

    seed = sys.argv[3]

    const_file = "default_consts"  # "default_consts"  # "single_agent_consts"
    consts_mod = importlib.import_module("params.consts."+const_file)
    print("const file", const_file)

    folder = "data/{}_{}_seed{}/".format(sa_file, scenario_file, str(seed))




    # === Save files ===
    # Save the current state of the implementation to the Subfolder
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
