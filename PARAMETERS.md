

## Parameters for the results in the publication

## Constant Parameters

We considered these parameters as constant and did not vary them. 

| Variable | Value | Note |
|-----|-----|-----|
| n_agents_arrival | 2 | |
| p_arrival | 40 | population at arrival |
| time_end | 1900 | A.D.|
| moving_radius_arrival | 1 | km; after arrival, settlers remain close to Anakena Beach |
| droughts_rano_raraku | [[800, 1200], [1570, 1720]] | start and end years of droughts at Rano Raraku |
| f_pi_well | 1 | relative farming productivity index of well suited cells|
| f_pi_poor | 0.05 | relative farming productivity index of poorly suited cells|
| garden_area_m2 | 1000 | size of a garden in m^2 |
| gridpoints_y | 50 | number of gridpoints covering the rectangular map |
| gridpoints_x | 75 | number of gridpoints covering the rectangular map |
| n_trees_arrival | 16000000.0 | nr of trees at arrival |
| t_pref_max | 0.8 | maximum tree preference |
| t_pref_min | 0.2 | minimum tree preference |
| p_splitting_agent | 12 | individuals of a new agent after it splits from an existing |
| p_remove_threshold | 6 | minimum population size of an agent, below this threshold the agent is removed |
| satisfaction_equ | 0.68844221 | Satisfaction index at which the mean growth rate is 1; adopted from [Puleston2017] |
| evaluation_thresholds | see below | thresholds for 1% penalt and 99% penalty for the evaluation criterion in each cell and category | 
| alpha | see below | weights of each category in the decision about moving to a new location |


#### Alpha

| Freshwater distance w | Geography g | Pop Dens pd | Trees tr | Farming availability f | 
 |-----|-----|-----|-----|-----|
| 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |

#### Evaluation Thresholds

| w01 | w99 | el01 | el99 | sl01 | sl99 | pd01 | pd99 | tr01 | tr99 | f01 | f99 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 2.8 (a) | 275.4 (b) | 0 | 300 | 0 | 7.5 | 0 | 300 | 0 | (c) | 0 | (d) |

(a):  (0.5 km)^2 / (pi * rad_raraku^2)
(b): (5 km)^2 / (pi * rad_raraku^2)
(c): ag.t_pref * t_req_pp * ag.p * satisfaction_equ
(d): (1-ag.t_pref) * f_req_pp * ag.p * satisfaction_equ




## Parameters for sensitivity analysis

We experimented with different settings for these values. Here we show the parameter values corresponding to the main result in the publications.
They are stored in file 
```
./params/sa/default 
```


| Variable | Value | Note |
|-----|-----|-----|
| t_req_pp | 10 | required trees per person per year (for t_pref = 1) |
| f_req_pp | 6.79 | required farmed gardens (weighted by farming productivity) per person (for t_pref = 0) |
| time_arrival | 800 | A.D.; arrival of settlers |
| max_p_growth_rate | 1.007 | maximum mean growth rate of an agent (for satisfaction = 1) |
| map_tree_pattern_condition | max_el: 450 | maximum elevation for trees in a cell |
|  | max_sl: 10 | maximum slope for trees in a cell |
|  | tree_decrease_lake_distance: 0 | decreas of tree density with area-weighted distance to freshwater lakes. (0 means uniform distribution) |


### Parameters for differrent scenarios

These parameters determine the specific scenario in our results:
- Aggregate
- Homogeneous
- Constrained
- Full

| Variable | Aggregate | Homogeneous | Constrained | Full | Description |
|-----|-----|-----|-----|-----|-----|
| n_agents_arrival | 1 | 2 | 2 | 2 |  number of initial agents
| p_split_threshold | infinity |  36 | 36 | 36 |  population size that triggers the splitting of an agent |
| r_t | infinity | infinity |  2 | 2 | km; radius for tree harvest |
| r_f |  infinity | infinity |  1 | 1 | km; radius for farming gardens |
| gamma | 0 | 0 | 0 |  20 | importance of penalties in decision making on new settlement location |
