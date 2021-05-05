

# Default parameter choices 
to reproduce the results in the associated publication.

## Parameters for differrent scenarios
In folder ```params/scenarios/```

We consider three scenarios:
- Unconstrained
- Partly Constrained
- Fully Constrained

| Variable |  Unconstrained | Partly Constrained | Fully Constrained | Description |
|-----|-----|-----|-----|-----|
| r_t |  infinity |  2 | 2 | km; radius for tree harvest |
| r_c |  infinity |  1 | 1 | km; radius for cultivating gardens |
| gamma | 0 | 0 |  20 | importance of location preferences for an agent's decision on a new settlement location |


## Parameters for sensitivity analysis

We experimented with different settings for these values. Here we show the parameter values corresponding to the main result in the publications.
They are stored in file 
```
./params/sa/default 
```


| Variable | Value | Note |
|-----|-----|-----|
| t_req_pp | 10 | required trees per person per year (for t_pref = 1) |
| c_req_pp | 6.79 | required cultivated gardens (weighted by arability) per person (for t_pref = 0) |
| droughts_rano_raraku | [[800, 1200], [1570, 1720]] | start and end years (A.D:) of droughts at Rano Raraku |
| map_tree_pattern_condition | max_el: 450 | maximum elevation for trees in a cell |
|  | max_sl: 10 | maximum slope for trees in a cell |
|  | tree_decrease_lake_distance: 0 | decrease of tree density with area-weighted distance to lakes. (0 means uniform distribution, >0 clustered trees around the lakes) |


## Constant Parameters
In file ```params/consts/const_default.py```

We considered these parameters as constant and did not vary them. 

| Variable | Value | Note |
|-----|-----|-----|
| n_agents_arrival | 2 | |
| p_arrival | 40 | population at arrival |
| time_arrival | 800 | A.D.; arrival of settlers |
| max_p_growth_rate | 1.007 | maximum mean growth rate of an agent (for satisfaction = 1) |
| time_end | 1900 | A.D.|
| moving_radius_arrival | 1 | km; after arrival, settlers remain close to Anakena Beach |
| arability_well | 1 | relative farming productivity index of well suited cells|
| arability_poor | 0.05 | relative farming productivity index of poorly suited cells|
| garden_area_m2 | 1000 | size of a garden in m^2 |
| gridpoints_y | 50 | number of gridpoints covering the rectangular map |
| gridpoints_x | 75 | number of gridpoints covering the rectangular map |
| n_trees_arrival | 16000000.0 | nr of trees at arrival |
| t_pref_max | 0.8 | maximum tree preference |
| t_pref_min | 0.2 | minimum tree preference |
| p_splitting_agent | 12 | individuals of a new agent after it splits from an existing |
| p_split_threshold |  36 | population size that triggers the splitting of an agent |
| p_remove_threshold | 6 | minimum population size of an agent, below this threshold the agent is removed |
| satisfaction_equ | 0.68844221 | Satisfaction index at which the mean growth rate is 1; adopted from [Puleston2017] |
| evaluation_thresholds | see below | thresholds for 1% penalty and 99% penalty for the location preferences in each cell and category | 
| alpha | see below | weights of each location preference in the decision about moving to a new location |


### Location Preferences 
#### Evaluation variables
| Freshwater distance w | Geography g | Population Density pd | Availability of Trees tr | Availability of highly arable gardens cu | 
 |-----|-----|-----|-----|-----|
increases with squared distance to a lake (divided by the lake area) | increases with elevation and slope | increases with higher population within r_c | increases with the number of trees within r_t | increases with the number of uncultivated, gardens (well-suited and to a much smaller degree poorly suited) within r_c
 |low values preferred |low values preferred  |low values preferred |high values preferred |high values preferred |
 | - | -  | - | cell excluded if minimum value (t99) not reached |cell excluded if minimum value not reached (cu99) |


#### Weights of different location preferences and their evaluation variables

Alpha

| Lake distance ld | Orography or | Population Density pd | Availability of Trees tr | Availability of highly arable gardens cu | 
 |-----|-----|-----|-----|-----|
| 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |

#### Penalty thresholds for different location preferences and their evaluation variables

| ld01 | ld99 | el01 | el99 | sl01 | sl99 | pd01 | pd99 | tr01 | tr99 | cu01 | cu99 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 2.8 (a) | 275.4 (b) | 0 | 300 | 0 | 7.5 | 0 | 300 | 0 | (c) | 0 | (d) |

(a): corresponds to ```(0.5 km)^2 / (pi * rad_raraku^2)```, i.e. the value 500m away from lake Rano Raraku

(b): corresponds to ```(5 km)^2 / (pi * rad_raraku^2)```,  i.e. the value 5km away from lake Rano Raraku

(c): varies in each time step and for each agent ag: 

    ag.t_pref * t_req_pp * ag.p * satisfaction_equ

(d): varies in each time step and for each agent 
    
    ag: (1-ag.t_pref) * c_req_pp * ag.p * satisfaction_equ


