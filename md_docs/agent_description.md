# Agents

Household agents located on the island with a specific population, resource related attributes and an update procedure for each year.

## State variables of the agent entity 
- Location (x, y, cell)
- Populatoin size p
- preference, t_pref, of resources tree over farming produce
- farming yield from occupied gardens and their farming productivity
- cut trees in the current year
- satisfaction with resource harvest.

## Processes
### __init__
Initiate the model by seting constants and parameters, createing the map,

### run
Run one simulation

Steps:
    - Initialise agents
    - Loop through each time step
        - Make one time step (update all agents sequentially)
        - check whether there is a drought of rano raraku
        

### init_agents
Initalise the n_agents_arrival agents. 
The simulation starts with two agents (with a total population of 40 individuals) settling in proximity to Anakena Beach in the North part of the island in the year $t_{0} = 800{\rm A.D.}$, following [Bahn2017].
We assume, they erect a settlement nearby within radius moving_radius_arrival

### observe
At the current time step, store agent's traits, state of the environment and aggregate variables for one time step

### step
Proceed one time step
Steps:
    - sort agents randomly
    - perform updates of each agent sequentially
### P_cat
Return the penalty following a logistic function of an evaluation variable for cells in a specified category.

Idea:
    For each evaluation category (${\rm cat}$) an agent defines a categorical penalty, $P_{\rm cat}(c)$, and evaluates all cells accordingly.
    The more unfavourable the conditions are in a cell, the higher the cell's penalties.
    The penalties, $P_{\rm cat}(c)$, depend logistically on the correlated, underlying geographic condition ranging from $0$ (very favourable condition) to $1$ (very unfavourable).
    The penalty is set to $\infty$ to inhibit a relocation to particular cells $c$, if the agent can not fill either its current tree or farming requirement for at least the upcoming year if it would move to this cell.

## Agent
Household agents located on the island with a specific population, resource related attributes and an update procedure for each year.

The (independent) state variables of the agent entity are:
- Location (x, y, cell)
- Populatoin size p
- preference t_pref of resources tree over farming produce
- farming yield from occupied gardens and their farming productivity
- cut trees
- satisfaction with resource harvest.

The processes are:
- calc_resource_req : calculate resource requirement for current year from constant tree/farming requirement per person farming/tree_req_pp, tree preference t_pref and population size p
- update_t_pref :  update the tree preference according to the level of deforestation in the local surrounding (with radius r_t).
- split_household : split household to form a new agent with population p_splitting_agent in a new location
- remove_agent : dissolve the agent, clear its land and remove all its individuals
- remove_unnecessary_gardens : if there is overproduce in farming, leave gardens (preferably poorly suited).
- population_change :  population growth/decline via a stochastic process for each individual to reproduce or die given the growth rate mu_mean from the past-dependent satisfaction index s
- mu_mean : return the mean growth rate given an agent's past-dependent satisfaction
- calc_penalty : calculate the penalty(ies) for cell(s) triangle_inds on the island based on weights on the specific evaluation criteria, alphas,and the current tree preference
- move : relocate the settlement according to stochastic process according to the total penalties of cells, within_inds
- occupy_gardens : occupy more gardens (preferably, well suited and with fewest trees needed to be cleared) in radius r_F until requirement f_req fulfilled or no further unoccupied gardens available in r_F.
- harvest_trees : cut trees in radius r_T until requirement fulfilled or no further trees available.
- update : update procedure of harvesting, population adaptation and potential moving for the agent in each time step

### __init__
Initialisation of the first settlers.

Agents of a model $m$ are initialised at a specific location $x$, $y$ on the model's discretised map of Easter Island and a poplation $p$
Other traits are model constants or derived from these constants and the location.

In detail, the parameters are:
\begin{tabular}{c|c|c|c}
    Variable & Description & Range & Units \\
    index & unique index/name & {0, 1, ...} & \\
    x, y & location of the agent on Easter Island & all set of values lying on cells of the EI map  & km \\
    cell & the index of the cell corresponding to (x,y) on the map & {0, 1, ...} \\
    p & population size & {p\_remove\_threshold, ..., p\_split\_threshold} & \\
    t_pref & tree preference & [t\_pref\_min, t\_pref\_max] (in [0,1]) & \\
    f_req & agent's required amount of farming produce (gardens times their farming_productivity) & [0,[ & gardens of 1000m2 with farming productivity 1 \\
    t_req & agent's required amount of trees per year & {0, ...} & trees \\
    tree_fill & fraction of tree requirement t_req, filled each year & [0,1] & \\
    farming_fill & fraction of farming requirement f_req filled by available gardens & [0,1] & \\
    satisfaction & Satisfaction with the resource situation in the current year. Minimum of tree_fill and farming_fill according to Liebig's law& [0,1] & \\
    past_satisfaction & past-dependent satisfaction with inertia. Average of this and past year's satisfaction & [0,1] & \\
    occupied_gardens_inds_map & indices of the cells in which a garden has been set up & list of values in self.m.map.inds_map & \\
    f_pi_occ_gardens & farming productivity of the cells of each garden in occupied_gardens_inds_map & list of values [0,1] & pp/garden \\
\end{tabular}

### calc_resource_req
calculate resource requirement for current year from constant tree/farming requirement per person farming/tree_req_pp, tree preference t_pref and population size p

For Trees:
$$ T_{\, \rm Req}^{\,\rm i}(t) =
    T_{\, \rm Pref}^{\, \rm i}(t) \cdot T_{\, \rm Req}^{\rm pP} \cdot p^{\,\rm i}(t)$$
For farmed gardens:
$$ F_{\, \rm Req}^{\,\rm i}(t) =
    (1-T_{\, \rm Pref}^{\, \rm i}(t)) \cdot F_{\, \rm Req}^{\,pP} \cdot p^{\,\rm i}(t) $$
     
### update_t_pref
 update the tree preference according to the level of deforestation in the local surrounding (with radius r_t).

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

### split_household
split household to form a new agent with population p_splitting_agent in a new location

### remove_unnecessary_gardens
if there is overproduce in farming, leave gardens (preferably poorly suited).
population growth/decline via a stochastic process for each individual to reproduce or die given the growth rate mu_mean from the past-dependent satisfaction index s
calculate the penalty(ies) for cell(s) triangle_inds on the island based on weights on the specific evaluation criteria, alphas,and the current tree preference
relocate the settlement according to stochastic process according to the total penalties of cells, within_inds
occupy more gardens (preferably, well suited and with fewest trees needed to be cleared) in radius r_F until requirement f_req fulfilled or no further unoccupied gardens available in r_F.
cut trees in radius r_T until requirement fulfilled or no further trees available.
update procedure of harvesting, population adaptation and potential moving for the agent in each time step

Initiated after population_change. A smaller population requires less farming produce and keeping the gardens is unnecessary for the agent for the moment.
As long as possible remove first the poorly suited, then well-suited gardens.

### remove_agent
dissolve the agent, clear its land and remove all its individuals

### population_change
population growth/decline via a stochastic process for each individual to reproduce or die given the growth rate mu_mean from the past-dependent satisfaction index s

The mean growth rate of the household $i$ is $\mu_{\rm m}^{\,\rm i}$, as a function of its past-dependent
satisfaction $S^{\,\rm i}$.
The growth/decline process is stochastic with each individual of the household having a probability
to die/reproduce of $|\mu_{\rm m}^{\,\rm i}(t)-1|$ in the corresponding regime in each time step.
Thus, on average a household grows with rate $\mu_{\rm m}^{\,\rm i}(t)$.
The characteristic value for a constant population size, $S_{\rm equ}$, is adopted from a demographic model
in \citet{Puleston2017}.

### mu_mean
Return the mean growth rate given an agent's past-dependent satisfaction

### calc_penalty
Calculate the penalty(ies) for cell(s) triangle_inds on the island based on weights on the specific evaluation criteria, alphas,and the current tree preference

#### Idea:
We make the following assumption.
Agents prefer a certain geography (low altitude and low slope), proximity to freshwater lakes
(weighted by the area), low population density, as well as large numbers of trees within $r_T distance
and high availability of arable (in particular, well-suited) potential gardens within $r_F$ distance.
The penalties, $P_{\rm cat}(c)$, depend logistically on the correlated, underlying geographic condition ranging from $0$ (very favourable condition) to $1$ (very unfavourable).
The penalty is set to $\infty$ to inhibit a relocation to particular cells $c$, if the agent can not fill either its current tree or farming requirement for at least the upcoming year if it would move to this cell.
All categorical penalties for a cell are then summed up using weights $\alpha_{\rm cat}$ (with $\sum_{\rm cat} \,  \alpha_{\rm cat}=1$) to obtain a total evaluation index of cell $c$'s suitability as a new household location.
Given the total penalty for each cell $c$, agent $i$ then moves to the cell $c$ with probability
$$ P_{\rm tot}(c) =  \sum_{\rm cat} \, \alpha_{\rm cat}^{\rm \, i}\cdot P_{\rm cat} \right) $$
The relative weight for farming, $\alpha_{\rm farming}$, and tree, $\alpha_{\rm tree}$, in equation \eqref{eq:p_Moving} differs between agents as we additionally scale them with the agent's current tree preference (and then re-normalise).
The other weights (for geography, freshwater proximity, and population density) remain the same for all agents.

##### More detailed calculation:

The total penalty for a cell is calculated as
$$ P_{\rm tot}^{\rm i}(c) = \sum_{\rm cat} \, \alpha_{\rm cat}^{\rm \, i} \cdot P_{\rm cat} $$
where $cat$ represents the categories used for the elevation of a cell:
- "w" for area wieghted proximity for freshwater, "pd" for population density, "tr" for tree availability,
- "f" for availability of well-suited and total gardens, "g" for geography (including elevation and slope)

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

#### Used for moving:
When an agent is prompted to move, the total penalty determines the probability of moving to a cell:
$$ p_{\rm m}^{\,\rm i}(c) =\frac{1}{\nu} \cdot \exp \left( - \gamma \cdot  P_{\rm tot}(c)$ \right) $$
where $\gamma$ is the factor determining how much the agent cares about the evaluation criteria
and $\nu$ is a normalisation.

### move 
relocate the settlement according to stochastic process according to the total penalties of cells, within_inds

#### Idea:
In our model, we allow agents to relocate their settlement on the island, when they split off from an existing agent or when they are sufficiently unsatisfied from the resource harvest, in particular, if both $S^{\,\rm i}(t) < S_{\rm equ}$ and current $s_{\rm curr}^{\,\rm i}(t) < S_{\rm equ}$.
When prompted to move, the agent decides on a new location by evaluating all cells on the island using several preferences and then assigning probabilities, accordingly.
This probabilistic approach accounts for the fact that human decision making is not simply a rational optimisation to find the best available location, but is e.g.\ limited by uncertainty and lack of knowledge or is based on heuristics rather than computation.
We assume that agents prefer a certain geography (low altitude and low slope), proximity to freshwater lakes, low population density, as well as large numbers of trees and high availability of arable (in particular, well-suited) potential gardens in the local surrounding.
Note that these preferences, are not related to the agent survival or its resource harvest.
For each of these categories (${\rm cat}$) the agent defines a categorical penalty, $P_{\rm cat}(c)$, and evaluates all cells accordingly.
The more unfavourable the conditions are in a cell, the higher the cell's penalties.

##### Steps:
- Clear your old space: remove gardens and decrease population of cell
- Evaluation of all possible cells and assignment of a total penalty
- Probability of moving to a cell from total penalty
- Moving:
    - Draw a new cell from probability distribution
    - If all penalties are infinite: choose random cell.
    - Draw a point in the triangle of the cell.


### occupy_gardens
occupy more gardens (preferably, well suited and with fewest trees needed to be cleared) in radius r_F until requirement f_req fulfilled or no further unoccupied gardens available in r_F.

#### Idea:
Agents occupy gardens of each 1000 m2 in arable cells
In each year all occupied gardens have a constant yield, given by the farming productivity index according to the classification of the corresponding cell into well or poorly suited for sweet potato cultivation, f_pi_occ_gardens
I.e. $$ F^{\,\rm i}(t) = \sum_{\, \rm g\,  \in  \, {\rm Gardens}^{\,\rm i}(t)}  \ F_{\, \rm P}(c_{\, \rm g}) $$
where ${\rm Gardens}^{\,\rm i}(t)$ are all $1000\, {\rm m^2}$ gardens farmed by the agent at time step $t$ in the corresponding cells $c_{\rm \, g}$.
If more farming produce is required, $F^{\,i}(t) \leq F_{\rm Req}^{\,\rm i}(t)$, the agent tries to occupy more prefereably well-suited gardens within $r_F$
Such potential garden areas might, however, still be forested.
Then, agents use slash-and-burn to clear the space and occupy an area of $A_{\rm \, garden} = 1000\, {\rm m^2}$.
Since we assume that trees are evenly spread within a cell, the fraction of removed trees is equivalent to the fraction of cleared area in this cell.
$$ A_\text{\, free}(c, t)\, [{\rm m^2}] = A_{\rm \, c} \cdot \frac{T(c, t_{0}) - T(c, t)}{T(c, t_{0})}$$
Some of that cleared space might already be occupied with existing gardens, $n_{\rm \, occ}(c,t)$.
$$ A_\text{occ}(c, t)\, [{\rm m^2}] = A_{\rm \, garden} \cdot n_{\rm \, occ}(c, t) $$
Hence, to occupy a new garden in cell $c$, the agent needs to clear trees until the condition
$$ A_\text{free}(c, t) - A_\text{occ}(c, t) \geq A_{\rm garden} $$
In our model, agents choose new gardens in well-suited cells one-by-one, beginning with the cell in which the least amount of trees needs to be cleared to obtain the required free space (optimally, there is already cleared space and no trees need to be removed additionally).
The addition of a garden immediately increases the agent's sweet potato yield $F^{\,i}(t)$.
Only when there are no more unfarmed, well-suited areas in the agent's surrounding, they also consider poorly suited cells (according to the same procedure).
This continues until $F^{\,i}(t) \geq F_{\rm Req}^{\,\rm i}(t)$, i.e.\ the requirement is filled, or no unfarmed, arable spaces remain within $r_{\rm F}$ distance of the agent.

##### Steps:
- Calculate farming_fill = fraction of required farming produce filled by current gardens and their specific
yields F_{PI}:
- If more gardens required:
    - determine neighbouring cells in r_F
    - determine well-suited cells in neighbours
    - do until satisfied or no well-suited, free gardens remain:
            - determine the cells with free, well-suited gardens,
            - determine the fraction of trees on the cell (assuming an initially uniform distribution within the cell)
            - determine the fraction of occupied gardens on the cell
            - determine how many trees need to be cleared to have free, cleared area sufficient for a further garden
            - select cell with the least amount of trees needing to be cleared
            - clear the necessary trees on that cell
            - occupy a garden on that cell and thereby increase current farming produce, i.e. append f_pi_occ_gardens and occupied_gardens_inds_map
    - repeat last two steps for poorly suited cells

### tree_harvest
Cut trees in radius r_T until requirement fulfilled or no further trees available.

##### Idea:
Individuals of the agent need a constant provision of trees (and their derivate products) in each year.

##### Steps:
- determine neighbouring cells in r_T
- calculate how many trees are reachable (in each cell)
- do until satisfied or no trees are available
    - select random cell with non-zero trees
    - remove one tree and thereby increase the own cut_trees

### update
update procedure of harvesting, population adaptation and potential moving for the agent in each time step

Specfic steps of the yearly update:
- Determine resource requirements (trees and farming)
- Try occupying more gardens until satisfied
- Try cutting trees until satisfied
- Determine new satisfaction index
- population growth, split or remove
- potentially move location
- update tree preference
