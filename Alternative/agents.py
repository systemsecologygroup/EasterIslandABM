import numpy as np
# import scipy.stats
from copy import copy


class Agent:
    """
    Household agents
    """

    def __init__(self, m, x, y, p):
        """

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
        self.m = m
        self.index = self.m.max_agent_index
        # already incremented in self.m.init_agents: self.m.max_agent_index += 1
        self.x = x
        self.y = y
        self.cell = -1  # of triangles on the map
        self.cell_all = -1  # of all triangles including ocean triangles
        self.p = p
        self.t_pref = self.m.t_pref_max
        self.f_req = 0
        self.t_req = 0
        self.tree_fill = 1
        self.farming_fill = 1
        self.satisfaction = 1
        self.past_satisfaction = 1
        self.occupied_gardens_inds_map = np.array([]).astype(np.int16)
        self.f_pi_occ_gardens = np.array([]).astype(np.float)
        # self.current_penalty

    def calc_resource_req(self):
        """
        Calculate the farming/tree requirement

        For Trees:
        $$ T_{\, \rm Req}^{\,\rm i}(t) =
            T_{\, \rm Pref}^{\, \rm i}(t) \cdot T_{\, \rm Req}^{\rm pP} \cdot p^{\,\rm i}(t)$$
        For farmed gardens:
        $$ F_{\, \rm Req}^{\,\rm i}(t) =
            (1-T_{\, \rm Pref}^{\, \rm i}(t)) \cdot F_{\, \rm Req}^{\,pP} \cdot p^{\,\rm i}(t) $$
        """

        self.f_req = (self.p * self.m.f_req_pp * (1 - self.t_pref))
        # TODO not needed self.f99 = (1 - self.T_Pref_i) * config.F_Req_pP * self.pop * config.S_equ
        self.t_req = (self.p * self.m.t_req_pp * self.t_pref)
        return

    def update_t_pref(self):
        """
        Update the tree preference according to local surrounding.

        Assume linear relation between agent $i$'s $T_{\rm Pref}^{\,\rm i}(t)$ and the level of deforestation
        $\epsilon$ within $r_{\rm T}$ distance:
        $$ \epsilon(t) =  \frac{\sum_{c \in C_{\rm T}(x^{\,\rm i}, y^{\,\rm i})} T(c, t)}
            {\sum_{c \in C_{\rm T}(x^{\,\rm i}, y^{\,\rm i})} T(c, t_{0})} $$
        where $C_{\rm T}(x^{\,\rm i}, y^{\,\rm i})$ is the circle around the agent with radius $r_{\rm T}$.
        Then, the updated tree preference of agent $i$ at time $t$ is
        $$ T_{\rm Pref}^{\,\rm i} (t) = T_{\rm Pref}^{\, \rm min} +
            \epsilon \cdot \left( T_{\rm Pref}^{\, \rm max} - T_{\rm Pref}^{\, \rm min}\right) \, ,$$
        where $T_{\rm Pref}^{\, \rm min/max}$ are lower and upper bounds to ensure that, on the one hand,
            an agent always requires some trees (e.g.\ for tools) and, on the other hand, some farming produce.

        """
        # level of deforestation around the agent
        epsilon = self.m.map.trees_map[self.m.map.circ_inds_trees[self.cell]].sum() / \
                  self.m.map.trees_cap[self.m.map.circ_inds_trees[self.cell]].sum()
        # linear increase:
        self.t_pref = epsilon * (self.m.t_pref_max - self.m.t_pref_min) + self.m.t_pref_min
        return


    def split_household(self):
        """
        split household and move new household

        If $p^{\, \rm i}(t)$ exceeds $p_{max}=36$ individuals , $p_{\rm split} = 12$ individuals split off
        to form a new agent in a new location.
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
        remove an number of gardens if the household does not require them anymore
            e.g. because the population number decreased

        as long as remove first the poorly suited, then well-suited
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
        remove agent if it becomes too small

        If $p^{\rm \, i}(t) < p_{\rm min} = 6$ individuals, the whole household dies and is removed.
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
        Population growth/decline according to happiness.

        The mean growth rate of the household $i$, $\mu_{\rm m}^{\,\rm i}$, as a function of its past-dependent
        satisfaction $S^{\,\rm i}$.
        The growth/decline process is stochastic with each individual of the household having a probability
        to die/reproduce of $|\mu_{\rm m}^{\,\rm i}(t)-1|$ in the corresponding regime in each time step.
        Thus, on average a household grows with rate $\mu_{\rm m}^{\,\rm i}(t)$.
        The characteristic value for a constant population size, $S_{\rm equ}$, is adopted from a demographic model
        in \citet{Puleston2017}.

        """
        # past-dependent satisfaction as average of current and last satisfaction value.
        past_dependent_satisfaction = 0.5 * (self.satisfaction + self.past_satisfaction)
        # mean growth rate
        mu_mean = self.m.mu_mean(past_dependent_satisfaction)

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


    def calc_penalty(self, triangle_inds):
        """
        calculate the total penalty of a number of cells for the agent.

        Assumption:
            Agents prefer a certain geography (low altitude and low slope), proximity to freshwater lakes
            (weighted by the area), low population density, as well as large numbers of trees within $r_T$ distance
            and high availability of arable (in particular, well-suited) potential gardens within $r_F$ distance.
        Calculation:
            $$ P_{\rm tot}^{\rm i}(c) = \sum_{\rm cat} \, \alpha_{\rm cat}^{\rm \, i} \cdot P_{\rm cat} $$
            where $cat$ represents the categories used for the elevation of a cell:
                "w" for area wieghted proximity for freshwater, "pd" for population density, "tr" for tree availability,
                "f" for availability of well-suited and total gardens, "g" for geography (including elevation and slope)
            The terms are explained in the following:
            - $\alpha_{\rm cat}^{\rm i} = \alpha_{\rm cat}$ for "w", "pd" and "g" are constant weights
            - $\alpha_{\rm tr}^{\rm i}$ and $\alpha_{\rm f}^{\rm i}$ are adjusted by the tree preference:
                With Normalisation factor
                $\eta^{\rm i} = \alpha_{\rm tr} \cdot T_{\rm pref}^{\rm i} + \alpha_{\rm f} \cdot (1-T_{\rm pref}^{\rm i})$
                the agent specific weights adjusted for the tree preference are
                $\alpha_{\rm tr}^{\rm i} = \frac{\alpha_{\rm tr} \cdot T_{\rm pref}^{\rm i} }{\eta^i}
                and
                $\alpha_{\rm f}^{\rm i} = \frac{\alpha_{\rm f} \cdot (1 - T_{\rm pref}^{\rm i})}{\eta^i}
            - $P_{\rm cat}(c)$ is the penalty assigned for a certain category given the value of the evaluation
                 criteria in the specific cell $c$.
                 The penalty grows logistically with $x$
                 $P_{\rm cat}(c) =  \frac{1}{1+exp[-k_x \cdot (x(c) - x_{0.5})]}$
                 - Here $x(c)$ is the value of an evaluation criteria for cell $c$ in the specific category $cat$:
                    - for "g": $P_g= 0.5 \cdot (P_{el} + P_{sl})$ with $x(c) = el(c)$ and $x(c) = sl(c)$
                    - for "w": $x(c) = min_{\rm lake} \left[ d_{\rm lake}^2 / A_{\rm lake} \right]$,
                        where $d_{\rm lake$ ist the distance and $A_{\rm lake}$ is the area of lakes Raraku, Kau, Aroi
                    - for "pd": $x(c) = \sum_{c' \in C_{\rm F}(c)} {\rm pop}(c') / (\sum_{c' \in C_{\rm F}(c)}  A_{\rm c)[km^2]$
                    - for "tr": $x(c) = \sum_{c' \in C_{\rm T}(c)} T(c')$
                        Additional: If $x(c) < tr99$, i.e.\ not even enough trees for the current requirement, then the
                        penalty is set to a value $>>>1$.
                    - for "f": $P_f= 0.5 \cdot (P_{f-tot} + P_{f-well})$
                        with $x_{f-tot}(c) = \sum_{c' \in C_{\rm F}(c)}  F_{\rm PI}(c') \cdot (n_{\rm \, gardens}(c') - n_{\rm \, occ}(c'))$
                        and $x_{f-well}(c) = \sum_{c' \in C_{\rm F}(c) {\rm \ with\ F_{\rm PI}(c')=1}  (n_{\rm \, gardens}(c') - n_{\rm \, occ}(c'))$
                        respectively.
                        Additional: If $x_{f-tot}(c) < f99$, i.e.\ not even enough gardens to farm for the current
                        requirement then the penalty is set to a value $>>>1$.

                 - $k_x$ is the steepness of the logistic function, which is determined by the evaluation_thresholds
                    for each category:
                        - $x_01$, the value at which the penalty is $P_{cat}|_{x=x01} = 0.01$
                        - and $x_99$, the value at which the penalty is $P_{cat}|_{x=x99} = 0.99$
                    Then $k_x = \frac{1}{0.5*(x99-x01)} \log(0.99/0.01)$
                - $x_{0.5} = 0.5\cdot (x01 + x99)$, the value of x at which the penalty is $0.5$

        When an agent is prompted to move, the total penalty determines the probabilit of moving to a cell:
        $$ p_{\rm m}^{\,\rm i}(c) =\frac{1}{\nu} \cdot \exp \left( - \gamma \cdot  P_{\rm tot}(c)$ \right) $$
        where $\gamma$ is the factor determining how much the agent cares about the evaluation criteria
        and $\nu$ is a normalisation.


        Parameters
        ----------
        triangle_inds : array of ints
            array of indices (on the island) on which to calculate the penalty

        Returns
        -------
        p_tot : array of floats between 0 and 1
            the weighted sum of all penalty(-ies) for the corresponding cells
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
        avail_farming_produce_cells = self.m.map.f_pi_c * (self.m.map.n_gardens_percell - self.m.map.occupied_gardens)
        f_tot = np.dot(avail_farming_produce_cells, circ_inds_farming_arr)
        p_f_tot = self.m.P_cat(f_tot, "f", infty_penalty="smaller than x99", ag=self)

        # for well suited cells only
        avail_well_farming_produce_cells = (self.m.map.f_pi_c == self.m.f_pi_well) * (self.m.map.n_gardens_percell - self.m.map.occupied_gardens)
        f_well = np.dot(avail_well_farming_produce_cells, circ_inds_farming_arr)
        p_f_well = self.m.P_cat(f_well, "f", infty_penalty="none", ag=self)

        # combined
        p_f = 0.5 * (p_f_tot + p_f_well)

        # === Tot Penalty ===
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
        return p_tot, [p_w, p_g, p_tr, p_f, p_pd]


    def move(self, within_inds):
        """
        move to a new location (in the triangles of within_inds) , after being prompted to do so.

        Steps:
            - Clear your old space: remove gardens and decrease population of cell
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

        # === Move ===
        if any(pr_c > 0):
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

    def update(self):
        """
        Update an agent:
            - Determine Resource requirements (trees and farming)
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

        # check if household is removed
        if self.p < self.m.p_remove_threshold:
            self.remove_agent()
            survived = False
        else:
            survived = True
            self.remove_unnecessary_gardens()

        # Strategy: split unhappy, move if household too small to fit.:

        # ALTERNATIVE IMPLEMENTATION:
        # So far:
        #   - split if household gets very big,
        #   - move if you are unhappy
        # Alternative:
        #   - split if unhappy, reduce to half the size (this is an adaptation)
        #   - move if hoseuhold is too small to split (this is the final possibility!)

        # if survived:
        #     if past_dependent_satisfaction < self.m.s_equ:
        #         if self.p > 2 * self.m.p_splitting_agent:
        #             init_p = copy(self.p)
        #             while True:  # self.2 * self.m.p_splitting_agent:
        #                 self.split_household()
        #                 if self.p < 0.5 * init_p:
        #                     break
        #         else:
        #             self.move(np.arange(len(self.m.map.inds_map)))
        #     self.update_t_pref()
        #     self.calc_resource_req()

        # Alternative 2:
        #   - split with prob 1-s (this is an adaptation)
        #   - move if hoseuhold is too small to split (this is the final possibility!)

        if survived:
            while np.random.random() < abs(1-past_dependent_satisfaction):
                if self.p > 2 * self.m.p_splitting_agent:
                    self.split_household()
                else:
                    self.move(np.arange(len(self.m.map.inds_map)))
            #if past_dependent_satisfaction < self.m.s_equ and self.satisfaction < self.m.s_equ:
            #    self.move(np.arange(len(self.m.map.inds_map)))
            self.update_t_pref()
            self.calc_resource_req()

        return


    def occupy_gardens(self):
        """
        If required, occupy (if available) more gardens in r_F distance

        Steps:
        - Calculate farming_fill = fraction of required farming produce filled by current gardens and their specific
        yields F_{PI}:
        - If more gardens required:
            - determine neighbouring cells in r_F
            - determine well-suited cells in neighbours
                Do until satisfied or no well-suited, free gardens remain:
                    - determine the cells with free, well-suited gardens,
                    - determine the fraction of trees on the cell (assuming an initially uniform distribution within the cell)
                    - determine the fraction of occupied gardens on the cell
                    - determine how many trees need to be cleared to have free, cleared area sufficient for a further garden
                    - select cell with the least amount of trees needing to be cleared
                    - clear the necessary trees on that cell
                    - occupy a garden on that cell and thereby increase current farming produce
                Repeat loop for poorly suited cells

        """
        self.farming_fill = (np.sum(self.f_pi_occ_gardens) / self.f_req).clip(max=1)

        if self.farming_fill < 1:
            # cell indices that are in r_F distance:
            circle_f_inds = np.where(self.m.map.circ_inds_farming[:, self.cell])[0]
            #
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
        harvest required amount of trees (as long as available)

        Steps:
        - determine neighbouring cells in r_T
        - calculate how many trees are reachable (in each cell)
        - do until satisfied or no trees are available
            - select random cell with non-zero trees
            - remove one tree and thereby increase the own cut_trees

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

        # Check
        if any(self.m.map.trees_map < 0):
            print("ERROR Trees <0")
            quit()

        return
