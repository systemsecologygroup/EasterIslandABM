# Model

<p align="center">
  <img src="../readme_pics/sketch.png" width="800">
</p>


## Short Summary:   
The ABM consists of multiple agents situated on a realistic representation of Easter Island's environment.
The environment is encoded on a 2D discretised map with heterogeneous geographic and
biological features. Agents represent households, who rely on two limited resources provided by this environment:
(1) Non-renewable palm trees (or derivate products like firewood, canoes or sugary sap e.g. [Bahn2017])
and (2) sweet potatoes. Agents obtain these resources by cutting trees and cultivating arable farming sites in
their near surroundings. Thereby, they change their local environment. The household's population growth or
decline consequently depends on the success of this resource acquisition. Furthermore, the resource availability
and other geographic indicators, like elevation or distance to freshwater lakes, determine the moving behaviour
of the agents on the island. The interaction with the natural environment, thus, constrains and shapes settlement
patterns as well as the dynamics of the population size of the Easter Island society.

## Time in the Model
The simulation starts with two agents (with a total population of 40 individuals) settling in proximity to Anakena Beach in the North part of the island in the year 800 A.D., following [Bahn2017].
All agents are then updated in yearly time steps up to 1900 A.D..
With the growing impacts through voyagers arriving on Easter Island in the 18th and 19th centuries, deportations of inhabitants as slaves and introduction of new diseases, the prehistoric phase and the island's isolated status end.

## Functions

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
    
    
