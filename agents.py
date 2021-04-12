'''
    File name: agents.py
    Author: Peter Steiglechner
    Date created: 01 December 2020
    Date last modified: 12 April 2021
    Python Version: 3.8
'''

import numpy as np
from copy import copy

class Agent:
    """
    Household agents located on the island with a specific population, resource related attributes and an update
    procedure for each year.
    They interact with the environment through harvesting trees and farming produce.

    Variables
    =========
    The (independent) state variables of the agent entity are:
    - Location (x, y, cell)
    - Population size p
    - preference t_pref of resources tree over farming produce
    - farming yield from occupied gardens (and their farming productivity)
    - cut trees in each year
    - satisfaction with resource harvest.

    Processes
    =========
    The processes are:
    - calc_resource_req
    - update_t_pref
    - split_household
    - remove_agent
    - remove_unnecessary_gardens
    - population_change
    - mu_mean
    - calc_penalty
    - move
    - occupy_gardens
    - harvest_trees
    - update
    """

    def __init__(self, m, x, y, p):
        """
        Initialisation of agents.

        The following table summarizes an agent's independent and major helping/dependent attributes:
        +===============+===============================================+===========================+===============+
        |   Variable    |   Description                                 |   Range                   |   Units       |
        +===============+===============================================+===========================+===============+
        | index         | unique index/name                             | {0, 1, ...}               |               |
        +---------------+-----------------------------------------------+---------------------------+---------------+
        | x, y          | location of the agent on Easter Island        | all positions on EI       | km            |
        +---------------+-----------------------------------------------+---------------------------+---------------+
        | cell          | the index of the cell corresp to (x,y) on EI  | {0, 1, ...}               |               |
        +---------------+-----------------------------------------------+---------------------------+---------------+
        | p             | population size                               | {p_remove_threshold, ..., | ppl           |
        |               |                                               |       p_split_threshold}  |               |
        +---------------+-----------------------------------------------+---------------------------+---------------+
        | t_pref        | tree preference                               | [t_pref_min, t_pref_max]  |               |
        +---------------+-----------------------------------------------+---------------------------+---------------+
        | f_req         | required amount of farming produce (gardens)  | [0,[                      | #gardens with |
        |               |                                               |                           | farm_prod=1   |
        +---------------+-----------------------------------------------+---------------------------+---------------+
        | t_req         | required amount of trees per year             | {0, ...}                  |               |
        +---------------+-----------------------------------------------+---------------------------+---------------+
        | tree_fill     | fraction of tree requirement t_req in a year  | [0,1]                     |               |
        +---------------+-----------------------------------------------+---------------------------+---------------+
        | farming_fill  | fraction of farming requirement f_req         | [0,1]                     |               |
        |               |                   filled by occupied gardens  |                           |               |
        +---------------+-----------------------------------------------+---------------------------+---------------+
        | satisfaction  | with resource fill in current year = Min of   | [0,1]                     |               |
        |               |    tree_fill and farming_fill ~ Liebig's law  |                           |               |
        +---------------+-----------------------------------------------+---------------------------+---------------+
        | past_satis-   | past-dependent satisfaction = average of      | [0,1]                     |               |
        |       faction |      this and past year's satisfaction        |                           |               |
        +---------------+-----------------------------------------------+---------------------------+---------------+
        | occupied_gar- | indices of the cells in which a garden is     | list of values in         |               |
        | dens_inds_map |          occupied by the agent                |      self.m.map.inds_map  |               |
        +---------------+-----------------------------------------------+---------------------------+---------------+
        | f_pi_occ_-    | farming productivity of the cells of each     | list of values [0,1]      |               |
        |   gardens     |  garden in occupied_gardens_inds_map          |                           |               |
        +===============+===============================================+===========================+===============+

        Parameters
        ----------
        m : instance of Model
            the model object hosting the agent.
        x: float
            x position in km on Easter Island
        y: float
            y position in km on Easter Island
        p: int
            initial population of the household
        """
        self.m = m  # Model
        self.index = self.m.max_agent_index  # running index giving each agent an unique index
        # incremented in the model: self.m.max_agent_index += 1
        self.x = x  # location
        self.y = y  # location
        self.cell = -1  # of triangles on the map
        self.cell_all = -1  # of all triangles including ocean triangles
        self.p = p  # population size
        self.t_pref = self.m.t_pref_max  # tree preference
        self.update_t_pref()
        self.f_req = 0  # agent's required amount of (gardens * farming_productivity)
        self.t_req = 0  # agent's required amount of trees per year
        self.tree_fill = 1  # fraction of tree requirement t_req filled each year
        self.farming_fill = 1  # fraction of farming requirement f_req filled by available gardens & [0,1] & \\
        self.satisfaction = 1  # satisfaction with harvest success index with
        self.past_satisfaction = 1  # satisfaction of previous time step
        self.occupied_gardens_inds_map = np.array([]).astype(np.int16)  # Indices of the cell (in self.m.map.inds_map) of each garden occupied
        self.f_pi_occ_gardens = np.array([]).astype(np.float)  # farming productivity index of each garden occupied
        return

    def calc_resource_req(self):
        """
        calculate resource (tree and farming) requirement for current year

        Uses the constant tree/farming requirement per person farming_req_pp/tree_req_pp, and the agent's tree
        preference t_pref and population size p
        """
        self.f_req = (self.p * self.m.f_req_pp * (1 - self.t_pref))
        self.t_req = (self.p * self.m.t_req_pp * self.t_pref)
        return

    def update_t_pref(self):
        """
        update the tree preference according to the level of deforestation in the local surrounding.

        Here, we assume a linear relation between the agent's tree preference and the level of deforestation (between 0
        and 100%) in its local surrounding (with radius r_t).
        Additionally the tree preference is bound by constants 0<t_pref_min <t_pref_max<1. 
        """
        # level of deforestation around the agent
        epsilon = self.m.map.trees_map[self.m.map.circ_inds_trees[self.cell]].sum() / \
                  self.m.map.trees_cap[self.m.map.circ_inds_trees[self.cell]].sum()
        # linear increase:
        self.t_pref = epsilon * (self.m.t_pref_max - self.m.t_pref_min) + self.m.t_pref_min
        return


    def split_household(self):
        """
        split household to form a new agent with population p_splitting_agent in a new location

        Process is initiated if population p exceeds p_max individuals.
        The new agent immediately moves away.
        """
        # reduce population by the splitting amount
        self.p -= int(self.m.p_splitting_agent)
        # resource requirements and t_pref are updated in the update function

        # Create child by copying the agent and updating its population, resource requirements, t_pref, ...
        child = copy(self)
        child.index = self.m.max_agent_index
        self.m.max_agent_index += 1
        child.p = int(self.m.p_splitting_agent)
        child.update_t_pref()
        child.calc_resource_req()
        child.occupied_gardens_inds_map = np.array([]).astype(int)
        child.f_pi_occ_gardens = np.array([])

        # Move the child agent and append to list of agents
        child.move(np.arange(len(self.m.map.inds_map)))
        self.m.schedule.append(child)
        return

    def remove_unnecessary_gardens(self):
        """
        remove gardens if there is too much overproduce given the population size and the household's required produce
            per person

        Initiated after population_change (and potentially split household).
        A smaller population requires less farming produce and keeping the gardens is unnecessary effort for the agent
        at least for the moment.
        As long as possible, remove first the poorly suited, then well-suited gardens.
        """
        f_overproduce = np.sum(self.f_pi_occ_gardens) - self.f_req
        # loop through gardens, ordered by their farming productivity from poor to well-suited
        for garden in self.occupied_gardens_inds_map[np.argsort(self.f_pi_occ_gardens)]:
            # try reducing the current garden and see
            f_overproduce -= self.m.map.f_pi_c[garden]
            if f_overproduce >= 0:
                # if there is still overproduce, remove the garden from the map
                self.m.map.occupied_gardens[garden] -= 1
                # remove the garden from the agent
                self.f_pi_occ_gardens = np.delete(self.f_pi_occ_gardens,
                                        np.where(self.occupied_gardens_inds_map == garden)[0][0])  # only delete first occurence
                self.occupied_gardens_inds_map = np.delete(self.occupied_gardens_inds_map,
                                       np.where(self.occupied_gardens_inds_map == garden)[0][0])  # only delete first occurence
            else:
                # if there is no overproduce, done
                return
        return


    def remove_agent(self):
        """
        dissolve the agent, clear its land and remove all its individuals.

        Initiated after population_change only when the population size has decreased to a value smaller than the
            pop_remove_threshold
        """
        # remove population from the cell
        self.m.map.pop_cell[self.cell] -= self.p
        # remove all gardens
        for garden in self.occupied_gardens_inds_map:
            self.m.map.occupied_gardens[garden] -= 1
        self.occupied_gardens_inds_map = np.array([])
        self.f_pi_occ_gardens = np.array([])
        self.m.schedule.remove(self)
        print("Agent ", self.index, " died.")
        self.m.excess_deaths += self.p
        self.p = 0
        return

    def population_change(self):
        """
        population growth/decline via a stochastic process for each individual

        The population growth/decline process is stochastic with each individual of the household having a probability
        to die or reproduce given by the mean growth rate mu_mean.
        Thus, on average a household grows with rate mu_mean.
        This mean growth rate is calculated from the past-dependent satisfaction index s in a separate function mu_mean.
        """
        # past-dependent satisfaction as average of current and last satisfaction value.
        past_dependent_satisfaction = 0.5 * (self.satisfaction + self.past_satisfaction)
        # mean growth rate
        mu_mean = self.mu_mean(past_dependent_satisfaction)

        # random values for each individual of the population
        rands = np.random.random(size=self.p)
        # get excess births if mu_mean>1, excess_deaths if mu_mean<1
        # number of excess births/deaths determined by random processes for each individual with prob. |mu_mean-1|
        excess_births = (mu_mean >= 1) * np.sum(rands < abs(mu_mean - 1))
        excess_deaths = (mu_mean <= 1) * np.sum(rands < abs(mu_mean - 1))
        self.p += excess_births - excess_deaths
        self.m.excess_deaths += excess_deaths
        self.m.excess_births += excess_births

        # adjust population in cell
        self.m.map.pop_cell[self.cell] += excess_births - excess_deaths
        return

    def mu_mean(self, s):
        """
        Return the mean growth rate given an agent's past-dependent satisfaction

        The characteristic value for a constant population size, s_equ, with mu(s_equ)=1 is adopted from a
        demographic model in Puleston et al. (2017).
        The max_p_growth_rate at s=1 is adopted from Bahn and Flenley (2017).

        Parameters
        ----------
        s : float between 0 and 1
            past dependent satisfaction of an agent
        """
        if s >= self.m.s_equ:
            m_grow = (self.m.max_p_growth_rate - 1) / (1 - self.m.s_equ)
            return m_grow * (s - self.m.s_equ) + 1
        else:
            m_decl = 1 / self.m.s_equ
            return m_decl * s


    def calc_penalty(self, triangle_inds):
        """
        Calculate the penalty(ies) for cell(s) triangle_inds on Easter Island based on weights, alpha, on the specific
        evaluation criteria and the current tree preference

        Idea and Assumptions
        ====================
        - Agents prefer a certain geography (low altitude and low slope), proximity to freshwater lakes
            (weighted by the area), low population density, as well as large numbers of trees within r_t distance
            and high availability of arable (in particular, well-suited) potential gardens within r_f distance.
        - The penalties, P_{cat}(c), depend logistically on correlated, underlying geographic condition variables
            ranging from 0 (very favourable condition) to 1 (very unfavourable).
        - The penalty is additionally set to infinity to inhibit a move to those cells c, in which the agent can not
            fill either its current tree or farming requirement for at least the upcoming year if it would move to this
            cell.
        - All categorical penalties for a cell are then summed up using (normed) weights alpha_cat to obtain a
            total evaluation index, P_tot, of cell c's suitability as a new household location.
            $$ P_{tot}(c) =  \sum_{cat} \alpha_{cat}^{i} \cdot P_{cat}(c) $$
            - The relative weights for farming, alpha_farming, and tree, alpha_tree, in the equation above vary for
                each agent as we additionally scale them with their current tree preference (and then re-normalise).
            -The other weights (for geography, freshwater proximity, and population density) remain const for all
                agents.

        More detailed calculation
        =========================
            The total penalty for a cell is calculated as
            $$ P_{tot}^{i}(c) = \sum_{cat} \alpha_{cat}^{i} \cdot P_{cat} $$
            where cat represents the categories used for the elevation of a cell:
            - "w" for area weighted proximity for freshwater,
            - "g" for geography (including elevation and slope)
            - "pd" for population density,
            - "tr" for tree availability,
            - "f" for availability of well-suited and total gardens,
            The weights alpha are explained in the following:
            - alpha_w, alpha_pd and alpha_g are constant weights
            - alpha_tr and alpha_f are multiplied by the agent's tree preference and then normalised such that sum of
                all alpha_cat is 1
            The penalties P_cat(c) is assigned for a certain category given the value of the evaluation criteria in the
                specific cell $c$:
                 - The penalty grows logistically with $x$
                    $$ P_{cat}(c) =  \frac{1}{1+exp[-k_x \cdot (x(c) - x_{0.5})]} $$
                 - Here x(c) is the value of an evaluation criteria for cell c in the specific category cat:
                    - for "g":
                        P_g= 0.5 \cdot (P_{el} + P_{sl}) with x(c)=el(c) and x(c) = sl(c)
                    - for "w":
                        x(c) = min_{lake} [ d_{lake}^2 / A_{lake} ],
                        where d_{lake} ist the distance and A_{lake} is the area of lakes Raraku, Kau, Aroi
                    - for "pd":
                        x(c) = \sum_{c' \in C_{F}(c)}  pop(c') / A_{C_F}
                        where pop(c) is the population in a cell, and A_{C_F} is the area for the cells in the
                            circle C_F around cell c
                    - for "tr":
                        x(c) = \sum_{c' \in C_{T}(c)} T(c')
                        Additional: If x(c) < tr99, i.e. there are not enough trees for the current requirement of the
                                agent, then the penalty is set to a value infinity.
                    - for "f":
                        P_f= 0.5 \cdot (P_{f-tot} + P_{f-well})
                        with
                            x_{f-tot}(c) = \sum_{c' \in C_{F}(c)}  F_{PI}(c') \cdot (n_{gardens}(c') - n_{occ}(c'))
                        and
                            x_{f-well}(c) = \sum_{c' \in C_{F}(c) with F_{PI}(c')=1}  (n_{gardens}(c') - n_{occ}(c'))
                        respectively.
                        Additional: If x_{f-tot}(c) < f99, i.e. there are not enough gardens for the current requirement
                            of the agent, then the penalty is set to a value infinity.

                 - k_x is the steepness of the logistic function, which is determined by the evaluation_thresholds
                    for each category:
                        - x_01, the value at which the penalty is P_{cat}|_{x=x01} = 0.01
                        - and x_99 the value at which the penalty is P_{cat}|_{x=x99} = 0.99
                    Then k_x = \frac{1}{0.5*(x99-x01)} log(0.99/0.01)
                - x_{0.5} = 0.5 \cdot (x01 + x99), the value of x at which the penalty is 0.5

        When ist the function called
        ============================
        When an agent is prompted to move, the total penalty is calculated for all cells.
        This total penalty for cells are reflected in the probability for the agent to move to a specific cell

        Parameters
        ----------
        triangle_inds : array of ints
            array of indices (on the island) on which to calculate the penalty

        Returns
        -------
        p_tot : array of floats between 0 and 1
            the weighted sum of all penalty(-ies) for the corresponding cells
        p_cat : list of arrays of floats between 0 and 1
            the penalty(-ies) for the corresponding cells in each category (w, g, tr, f, pd)
        """
        if self.m.gamma == 0:
            # Do not need to calculate the penalties, because they are not taken into account
            return np.zeros_like(triangle_inds), [np.zeros_like(triangle_inds) for _ in range(5)]

        # Boolean matrix of size len(inds_map) and len(triangle_inds).
        # c_ij = True if the cell with index i (in inds_map) is in r_t/f distance of cell j in triangle_inds
        circ_inds_trees_arr = (self.m.map.circ_inds_trees[:, triangle_inds]).astype(np.int)
        circ_inds_farming_arr = (self.m.map.circ_inds_farming[:, triangle_inds]).astype(np.int)

        # == Calculate each categories penalty ==

        # === Map/Geography Penalty ===
        p_g = self.m.map.penalty_g[triangle_inds]

        # === Tree Penalty ===
        tr = np.dot(self.m.map.trees_map, circ_inds_trees_arr)
        p_tr = self.m.P_cat(tr, "tr", infty_penalty="smaller than x99",  ag=self)

        # === Pop Density Penalty ===
        # Population density = sum of population in each cell in r_F / (nr of cells in r_F distance * cell area)
        pd = np.dot(self.m.map.pop_cell, circ_inds_farming_arr) / \
             (np.sum(circ_inds_farming_arr, axis=0) * self.m.map.triangle_area_m2 * 1e-6)
        p_pd = self.m.P_cat(pd, "pd")

        # === Freshwater Penalty ===
        p_w = self.m.map.penalty_w[triangle_inds]

        # === Agriculture Penalty ===
        # for poorly and well suited cells
        avail_garden_areas = (self.m.map.n_gardens_percell - self.m.map.occupied_gardens)
        avail_farming_produce_cells = self.m.map.f_pi_c * avail_garden_areas
        f_tot = np.dot(avail_farming_produce_cells, circ_inds_farming_arr)
        p_f_tot = self.m.P_cat(f_tot, "f", infty_penalty="smaller than x99", ag=self)

        # for well suited cells only
        avail_well_farming_produce_cells = (self.m.map.f_pi_c == self.m.f_pi_well) * avail_garden_areas
        f_well = np.dot(avail_well_farming_produce_cells, circ_inds_farming_arr)
        p_f_well = self.m.P_cat(f_well, "f", infty_penalty="none", ag=self)

        # combined
        p_f = 0.5 * (p_f_tot + p_f_well)

        # === Total Penalty ===
        # determine alpha_f and alpha_tr from the current tree preference of the agent
        eta = (self.t_pref * self.m.alpha["tr"] + (1 - self.t_pref) * self.m.alpha["f"]) / (
                self.m.alpha["tr"] + self.m.alpha["f"])
        alpha_tr = self.m.alpha["tr"] * self.t_pref / eta
        alpha_f = self.m.alpha["f"] * (1 - self.t_pref) / eta

        # linear combination of all weighted penalties for all cells of triangle_inds
        p_tot = (self.m.alpha["w"] * p_w +
                 alpha_tr * p_tr +
                 self.m.alpha["pd"] * p_pd +
                 alpha_f * p_f +
                 self.m.alpha["g"] * p_g
                 )
        p_cat = [p_w, p_g, p_tr, p_f, p_pd]
        return p_tot, p_cat


    def move(self, within_inds):
        """
        relocate the settlement according to stochastic process according to the total penalties of cells, within_inds

        Idea
        ====
        We allow agents to relocate their settlement on the island
            - when they split off from an existing agent or
            - when they are sufficiently unsatisfied from the resource harvest, in particular, if both
                past_satisfaction and current satisfaction are lower than the equilibrium satisfaction
        When prompted to move, the agent decides on a new location by evaluating all cells on the island using several
        preferences and then assigning probabilities, accordingly.
        This probabilistic approach accounts for the fact that human decision making is not simply a rational
        optimisation to find the best available location.
        Instead it is a process e.g. limited by uncertainty and lack of knowledge and thus based on heuristics rather
         than computation.
        The probabilities depend on a penalty evaluation described in detail in process calc_penalty.
        In a nutshell, we assume that agents prefer a certain geography (low altitude and low slope), proximity to
        freshwater lakes, low population density, as well as large numbers of trees and high availability of arable
        (in particular, well-suited) potential gardens in the local surrounding.
        Note that these preferences, are not related to the agent survival or its resource harvest.
        For each of these categories (cat) the agent defines a categorical penalty, P_{cat}(c), and evaluates all cells
         accordingly.
        The more unfavourable the conditions are in a cell, the higher the cell's penalties.
        Then the probability to move to a cell depends indirectly proportionally on the weighted total summed penalty.

        Steps
        =====
        - Clear your old space: remove gardens and decrease population of the cell
        - Evaluation of all possible cells and assignment of a total penalty
        - Probability of moving to a cell from total penalty
        - Moving:
            - Draw a new cell from probability distribution
            - If all penalties are infinite: choose random cell.
            - Draw a point in the triangle of the cell.

        Parameters
        ----------
        within_inds : array of ints
            inidices of the triangles on the map to which the agent can move

        """

        # === Clear your old space ===
        self.m.map.pop_cell[self.cell] -= self.p
        for garden in self.occupied_gardens_inds_map:
            self.m.map.occupied_gardens[garden] -= 1
        self.occupied_gardens_inds_map = np.array([]).astype(int)
        self.f_pi_occ_gardens = np.array([])

        # === Penalty Evaluation ===
        p_tot, [p_w, p_g, p_tr, p_f, p_pd] = self.calc_penalty(within_inds)

        # === Probabilities ===
        pr_c = np.exp( - self.m.gamma * p_tot)

        # PLOT THE PENALTIES AT TWO DIFFERENT SNAPSHOTS
        # if self.m.time == 1450 or self.m.time == 1500:
        #    for v, label in zip([p_tr, p_f, p_pd, p_tot], ["tr", "f", "pd", "tot"]):
        #        l = r"Penalty $P_{" + label + r"}"
        #        plot_map(self.m.map, v, l, cmapPenalty, 0, 1, str(self.m.time)+"_penalty_"+label, t=self.m.time)
        #    plot_map(self.m.map, pr_c, "Moving Probability", cmapProb, 1e-5, 1, str(self.m.time)+"_prob", t=self.m.time)

        # === Move ===
        if any(p_tot < 1):
            # if there is a cell with finite penalty (non-zero probability).
            pr_c *= 1/np.sum(pr_c)
            self.cell = np.random.choice(within_inds, p=pr_c)
        else:
            # Choose new cell randomly
            self.cell = np.random.choice(within_inds)
        self.cell_all = self.m.map.inds_map[self.cell]

        # Increase population in the cell.
        self.m.map.pop_cell[self.cell] += self.p

        # Find settlement location in the triangle:
        # Get corner points of the agent's chosen triangle
        corner_inds_new_cell = self.m.map.triobject.triangles[self.cell_all]
        corner_A, corner_B, corner_C = [[self.m.map.triobject.x[k], self.m.map.triobject.y[k]] for k in corner_inds_new_cell]
        # two random numbers s and t
        s, t = sorted([np.random.random(), np.random.random()])
        # Via barithmetric points.
        self.x = s * corner_A[0] + (t - s) * corner_B[0] + (1 - t) * corner_C[0]
        self.y = s * corner_A[1] + (t - s) * corner_B[1] + (1 - t) * corner_C[1]

        # Check:
        if not self.cell_all == self.m.map.triobject.get_trifinder()(self.x, self.y):
            print("Error in moving to the new triangle: Chosen location cell is not in chosen cell!")

        return


    def occupy_gardens(self):
        """
        occupy more gardens (preferably, well suited and with fewest number of trees needed to be cleared) in radius r_f
            until requirement f_req fulfilled or no further unoccupied gardens available in r_f.

        Idea
        ====
        - Agents have occupied gardens of each 1000 m2 in arable cells
            In each year, all occupied gardens have a constant yield, given by the farming productivity index according
            to the classification of the corresponding cell into well or poorly suited for sweet potato cultivation,
            f_pi_occ_gardens.
            I.e. the agent's obtained farming produce is the sum of all occupied gardens weighted by their cell's
            productivity indices.
        - If more farming produce is required, the agent tries to occupy more (prefereably well-suited) gardens within
            r_f
        - Such potential garden areas might, however, still be forested.
            Then, agents use slash-and-burn to clear the space and occupy an area of 1000 m^2 in that cell.
            We assume that trees are evenly spread within a cell, the fraction of removed trees is equivalent to the
            fraction of cleared area in this cell.
        - Some of that cleared space might already be occupied with existing gardens.
        - Hence, to occupy a new garden in a cell, the agent needs to clear trees until enough space for an additional
            garden is cleared
            I.e.:
            $$ A_{free}(c, t) - A_{occ}(c, t) >= A_{garden} = 1000 m^2 $$
            with A_{free}(c, t) the cleared area and A_{occ}(c, t) the area occupied by already existing gardens in cell
            c at time t
        - In our model, agents choose new gardens in well-suited cells one-by-one, beginning with the cell in which the
            least amount of trees needs to be cleared to obtain the required free space
            (optimally, there is already cleared space and no trees need to be removed additionally).
        - The addition of a garden immediately increases the agent's sweet potato yield
        - Only when there are no more unfarmed, well-suited areas in the agent's surrounding, they also consider poorly
            suited cells (according to the same procedure).
            This continues until the requirement is filled, or no unfarmed, arable spaces remain within r_f distance
                of the agent.

        Steps
        =====
        - Calculate farming_fill = fraction of required farming produce filled by current gardens and their specific
        yields F_{PI}:
        - If more gardens required:
            - determine neighbouring cells in r_f distance
            - determine well-suited cells
            - do until satisfied or no well-suited, free gardens remain:
                    - determine the cells with free, well-suited gardens,
                    - determine the fraction of trees on the cell (assuming an initially uniform distribution of trees
                            within the cell)
                    - determine the fraction of occupied gardens on the cell
                    - determine how many trees need to be cleared to have free, cleared area sufficient for setting up
                            a further garden
                    - select cell with the least amount of trees needing to be cleared
                    - clear the necessary trees on that cell
                    - occupy a garden on that cell and thereby increase the current farming produce, i.e. append
                        f_pi_occ_gardens and occupied_gardens_inds_map
            - repeat last two steps for poorly suited cells
        """
        self.farming_fill = (np.sum(self.f_pi_occ_gardens) / self.f_req).clip(max=1)

        if self.farming_fill < 1:
            # cell indices that are in r_F distance:
            circle_f_inds = np.where(self.m.map.circ_inds_farming[:, self.cell])[0]
            # first consider only well suited cells, then consider poorly suited cells.
            for gardens in [self.m.map.avail_well_gardens, self.m.map.avail_poor_gardens]:
                while True:
                    if self.farming_fill == 1:
                        break
                    # Cells within circle_f_inds, that are arable and have not yet all of its gardens occupied.
                    open_spaces = np.where((gardens[circle_f_inds] > 0) * (gardens[circle_f_inds] > self.m.map.occupied_gardens[circle_f_inds]))

                    # indices of the cells
                    potential_cells = circle_f_inds[open_spaces]
                    if len(potential_cells) == 0:
                        # no unoccupied gardens left in distance r_F around agent
                        break

                    # Fraction of cell occupied by trees on each potential cell
                    frac_cell_with_trees = self.m.map.trees_map[potential_cells] / (
                    self.m.map.trees_cap[potential_cells]).clip(1e-5)
                    # Fraction of cell that would be occupied by one garden
                    frac_cell_with_1garden = self.m.garden_area_m2 / self.m.map.triangle_area_m2
                    # Fraction of cell that is currently occupied by gardens
                    frac_cell_with_occ_gardens = frac_cell_with_1garden * self.m.map.occupied_gardens[potential_cells]
                    # Fraction of cell without gardens and without trees
                    frac_cell_notree_nogarden = (1 - frac_cell_with_trees) - frac_cell_with_occ_gardens

                    # Conditon: frac_cell_notree_nogarden >= frac_cell_with_1garden
                    # $A_{\rm free}(c, t) - A_{\rm occ}(c, t) \geq A_{\rm garden}$

                    # need to clear trees:
                    # $\Delta T =  T(c,0) \cdot (A_{garden} - (A_{free} - A_{occ}))/ A_{cell}$
                    # if frac_cell_with_1garden - frac_cell_notree_nogarden <= 0: condition fulfilled: No need to clear

                    trees_to_clear = self.m.map.trees_cap[potential_cells] * (
                        (frac_cell_with_1garden - frac_cell_notree_nogarden).clip(min=0))

                    # For some cells, we would need to clear more trees than there are left.
                    allowed_cells = np.where(trees_to_clear <= self.m.map.trees_map[potential_cells])[0]
                    if len(allowed_cells) == 0:
                        break

                    # Choose cell with the lowest amount of trees that need to be cleared
                    cell_with_min_clearance = allowed_cells[np.argmin(trees_to_clear[allowed_cells])]
                    index_of_chosen_cell = potential_cells[cell_with_min_clearance]

                    # Clear the trees there
                    trees_to_clear_in_cell = np.ceil(trees_to_clear[cell_with_min_clearance])
                    self.m.map.tree_clearance[index_of_chosen_cell] += trees_to_clear_in_cell
                    self.m.map.trees_map[index_of_chosen_cell] -= trees_to_clear_in_cell

                    # Occupy garden in the cell.
                    self.occupied_gardens_inds_map = np.append(self.occupied_gardens_inds_map,
                                                               index_of_chosen_cell).astype(int)
                    self.f_pi_occ_gardens = np.append(self.f_pi_occ_gardens, self.m.map.f_pi_c[index_of_chosen_cell])
                    self.m.map.occupied_gardens[index_of_chosen_cell] += 1

                    self.farming_fill = (np.sum(self.f_pi_occ_gardens) / self.f_req).clip(max=1)
        return

    def tree_harvest(self):
        """
        Cut trees in radius r_t until requirement fulfilled or no further trees available.

        Idea
        ====
        Individuals of the agent need a constant provision of trees (and their derivate products) in each year.

        Steps
        =====
        - determine neighbouring cells in r_t
        - calculate how many trees are present (in each cell)
        - do until satisfied or no trees are available:
            - select random cell with non-zero trees
            - remove one tree and, thereby, increase the own cut_trees
        """

        # cell indices that are in r_T distance:
        circle_t_inds = np.where(self.m.map.circ_inds_trees[:, self.cell])[0]
        # helper variable: tree numbers in all reachable cells
        reachable_trees_cells = self.m.map.trees_map[circle_t_inds]
        # total trees reachable
        n_reachable_trees = np.sum(reachable_trees_cells)

        # how many trees cut so far
        cut_trees = 0
        self.tree_fill = cut_trees / self.t_req
        while True:
            if self.tree_fill >= 1:
                # tree requirement fulfilled; break
                self.tree_fill = 1
                break
            if n_reachable_trees == 0:
                self.tree_fill = cut_trees / self.t_req
                break

            reachable_cell_ind = np.random.choice(np.where(reachable_trees_cells > 0)[0])
            cell_ind = circle_t_inds[reachable_cell_ind]

            # remove tree from helper variables and map
            reachable_trees_cells[reachable_cell_ind] -= 1
            n_reachable_trees -= 1
            self.m.map.trees_map[cell_ind] -= 1

            # Increment cut trees and adjust tree_fill
            cut_trees += 1
            self.tree_fill = cut_trees / self.t_req
        return

    def update(self):
        """
        update agent in each time step: harvest, change population (potentially splitting or removing the agent),
        potentially move the settlement, adapt tree preference

        Specfic steps of the yearly update:
            - Determine resource requirements (trees and farming)
            - Try occupying more gardens until satisfied
            - Try cutting trees until satisfied
            - Determine new satisfaction index
            - population growth, split or remove
            - potentially move location
            - update tree preference
        """

        # === Determine Resource requirements (trees and farming) ===
        self.calc_resource_req()

        # === Try occupying more gardens until satisfied ==
        self.occupy_gardens()

        # === Tree Harvest ===

        self.tree_harvest()

        # === Population Change ===

        self.past_satisfaction = copy(self.satisfaction)
        self.satisfaction = np.min([self.tree_fill, self.farming_fill])

        past_dependent_satisfaction = 0.5 * (self.past_satisfaction + self.satisfaction)
        self.population_change()

        # Strategy: First split if household becomes too big, then move if unhappy:

        # Check if household splits
        if self.p > self.m.p_split_threshold:
            self.split_household()

        # check if household is removed
        if self.p < self.m.p_remove_threshold:
            self.remove_agent()
            survived = False
        else:
            survived = True
            self.remove_unnecessary_gardens()

        # === Move ===
        if survived:
            self.calc_resource_req()
            if past_dependent_satisfaction < self.m.s_equ and self.satisfaction < self.m.s_equ:
                self.move(np.arange(len(self.m.map.inds_map)))
                self.m.resource_motivated_moves += 1
            else:
                # could calculate penalty here.
                pass

            # === Update tree preference ===
            self.update_t_pref()
            self.calc_resource_req()

        return



