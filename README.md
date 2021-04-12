# Easter Island ABM
 An Agent-Based Model (ABM) that simulates the spatial and temporal dynamics of household agents on Easter Island
    and their interactions with the natural environment through resource consumption prior to European arrival.

## Model
<p align="center">
  <img src="readme_pics/sketch.png" width="800">
</p>

###### Short Summary:   
The ABM consists of multiple agents situated on a realistic representation of Easter Island's environment.
The environment is encoded on a 2D discretised map with heterogeneous geographic and
biological features. Agents represent households, who rely on two limited resources provided by this environment:
(1) Non-renewable palm trees (or derivate products like firewood, canoes or sugary sap e.g. [Bahn2017])
and (2) sweet potatoes. Agents obtain these resources by cutting trees and cultivating arable farming sites in
their near surroundings. Thereby, they change their local environment. The household's population growth or
decline consequently depends on the success of this resource acquisition.
We define three adaptation mechanisms which we implement as heuristic rules for the agents:
First, agents split into two households when their population exceeds a threshold.
Second, failure to harvest a sufficient fraction of the required resources pushes agents to move their settlement.
Heterogeneous resource availability and other geographic indicators, like elevation or distance to freshwater lakes,
influence the decision on an agent's new location on the island.
Thirdly, agents adapt their preference for trees over sweet potatoes in response to a changing environment.
In summary, the interaction with the natural environment, constrains and shapes settlement and harvest patterns as
well as the dynamics of the population size of the Easter Island society.

###### Time in the Model
The simulation starts with two agents (with a total population of 40 individuals) settling in proximity to
        Anakena Beach in the North part of the island in the year t_0 = 800 A.D., following [Bahn2017].
All agents are then updated in yearly time steps up to 1900 A.D..
With the growing impacts through voyagers arriving on Easter Island in the 18th and 19th
        centuries, deportations of inhabitants as slaves and introduction of new diseases, the prehistoric
        phase and the island's isolated status end.

### Environment
Discretised representation of the map of Easter Island.
Each cell contains information about geographic properties like elevation or freshwater proximity 
and resource related variables like availability of trees, farming productivity of the soil and available space 
for setting up gardens.


<p align="center">
  <img src="readme_pics/F_P.png" width="300">
  <img src="readme_pics/Trees.png" width="300">
</p>

##### Some crucial properties for each cell
- constant farming productivity index (left panel of Figure)
- number of available well and poorly suited gardens
- number of available trees (right panel at time 800 A.D.)


### Agents
Household agents located on the island with a specific population size, resource related attributes and an update procedure for each year.

###### State variables of the agent entity 
- Location (x, y, cell)
- Populatoin size p
- preference, t_pref, of resources tree over farming produce
- farming yield from occupied gardens and their farming productivity
- cut trees in the current year
- satisfaction with resource harvest.


## How to run the model
### Single Run
Run the main python script and provide the corresponding, predefined parameter files in folder ```./params```.
These files contain dictionaries with the parameters for each specific experiment and scenario. 

- the first additional argument is the filename of parameter values for the different experiments tested in the 
    sensitivity analysis (folder params/sa/...):
        e.g. use ```default``` for /params/sa/default
- the second additional argument is the filename of parameter values for the specific scenario `homogeneous`, 
    `constrained`, or `full` (folder params/scenarios/...):
        e.g. use ```full``` for /params/sa/full
- the third additional argument denotes the (integer) seed value used.

For a detailed description of the parameters, look at [Parameters]{PARAMETERS.md}.
```
python main.py default full 1
```
Each simulation run will take about 5-10 minutes on a standard computer. 

### Ensemble Runs
Alternatively, to get the main results for all three scenarios `homogeneous', 
    `constrained', or `full' with multiple seeds as used for our results section, 
Run the following command e.g.\ on a cluster
```
./run_scenarios.sh default
```
and unpack later on local machine
```
./unpack.sh data/packed/default_full_seed
```
Run the analysis like 
```
cd plot_functions
./plot_analysis.sh default 10
```
("default" is the model experiment/sensitivity setting, "10" is the y_max of the island-wide population axis):

## Files
- ```agents.py ```
    contains the Agent class
- ```main.py ```
    contains the Model class
- ```create_map.py```
    contains the class Map defining a discretised environment
- ```saving.py ```
    contains helper functions to save the model's state
- ```./Map/```
    This folder contains the maps used as inputs in the creation of the discretised map:
    - ```elevation_EI.tif```
    - ```slope_EI.tif```
    - ```puleston2017_original.jpg```
- ```./plot_functions```
    This folder contains some scripts to reproduce the figures in the publication.
- ```./params```
    This folder contains python scripts that define dictionaries of parameters for 
    - the constant parameters, 
    - the ones coresponding to the senstivity analysis in the publication and
    - the scenarios presented in the main result section.
- ```run_scenarios.sh```
    Bash script to run all three scenarios at the same time
- ```run_ensemble.sh```
    Bash script to run all three scenarios at the same time
    
## Python Libraries and Dependencies

| Package  | Version |
|-----|-----|
| python | 3.8 |
| xarray | 0.16.1|
| scipy | 1.5.0 |
| numpy | 1.18.5 |
| matplotlib | 3.2.2 |
| pathlib | 1.0.1 |


## Further Reading

[Bahn2017] Bahn P, Flenley J (2017) Easter Island, Earth Island: The enigmas of Rapa Nui, 4th
edn. Rowman & Littlefield, Maryland, USA

[Puleston2017] Puleston CO, Ladefoged TN, Haoa S, Chadwick OA, Vitousek PM, Stevenson CM
(2017) Rain, sun, soil, and sweat: A consideration of population limits on Rapa Nui (Easter Island) before European contact. Frontiers in Ecology and Evolution
DOI 10.3389/fevo.2017.00069

[Rull2020] Rull, V. (2020), The deforestation of Easter Island. Biol Rev, 95: 124-141. https://doi.org/10.1111/brv.12556

# Author
Peter Steiglechner, April 2021. [Orcid Link]{https://orcid.org/0000-0002-1937-5983}

