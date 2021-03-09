# Map

Discretised representation of the map of Easter Island.

<p align="center">
  <img src="../readme_pics/F_P.png" width="300">
  <img src="../readme_pics/Trees.png" width="300">
</p>

## Idea:
Obtain an elevation map of a rectangular extract of Easter Island. Each pixel represents the elevation at the particular point.
Here, we obtain this from Google Earth Engine.
Define a grid of points within the map.
Define triangular cells in this grid.
Each cell is a microscopic unit of the discretised representation.
A cell has the following constant properties:
- elevation
- slope
- corresponding geography penalty.
- midpoint
- Area
- farming productivity index
- number of available well and poorly suited gardens
- tree carrying capacity (or trees at time of arrival)

A cell has additionally dynamic properties:
- trees
- area-weighted distance to freshwater lake / water penalty which depends on the droughts
- population
- number of trees cleared
- number of occupied gardens

additionally a cell can be:
- on land or ocean
- the cell of anakena beach,
- part of a freshwater lake
- a coastal cell


## Implementation details:
Static Variables:
- triobject : consisting of the matplotlib.triangulation object
    - x : coordinate of points in km
    - y : coordinate of points in km
    - triangles : the indices of the three corner points for each triangular cell
    - mask : denoting those triangles not on the ocean, i.e.\ where the midpoint has a non-zero elevation
- el_map : elevation on each cell in m above sea level
- sl_map : map of slope on each cell in degrees
- f_pi_c : the farming productivity index (f_pi_well for well suited and f_pi_poor for poorly suited, 0 else) for each cell
- trees_cap : number of trees in each cell before arrival of the first settlers, thus carrying capacity
- avail_well_gardens : Number of potential well-suited gardens in each cell determined by the area of a cell and the size of a garden
- avail_poor_gardens : Number of potential well-suited gardens in each cell determined by the area of a cell and the size of a garden

Dynamic Variables:
- water_cells_map : indices of cells containing water
- penalty_w : Penalty [0,1] for each cell depending on the distance to freshwater lakes, which depends on wether Rano Raraku is dried out.
- occupied_gardens : dynamic number of occupied gardens in each cell
- pop_cell : population in each cell
- tree_clearance : number of cleared trees in each cell
- trees_map : number of available trees in each cell.

The class needs:
- an elevation image (with a latitude and longitude bounding box)
- a slope image (with the same bounding box)
- Figure 4 of [Puleston2017], giving farming producitivity indices (with a longitude and latitude bounding box)


## Functions 

### __init__
create the discretised representation of Easter Island and assign values for all variables in all cells in numpy arrays.

#### Steps:
- Create a discretised representation of Easter Island via triangular cells
- calculate the distance matrix of all land cells
- determine cells that are at the coast and the distance of all land cells to the nearest coast.
- determine cell of Anakena Beach (on land and in total) and the cells within the initial moving radius
- get water cells for the Rano Aroi, Kau and Raraku and determine the cells, water penalties and distance to water in the peridos when Raraku is dried out and when it is not.
- calculate penalty of elevation and slope and combine them to geography penalty
- get the farming productivity indices and the amount of available farming gardens in each cell
- distribute trees on the island as the tree carrying capacity
- Initialise arrays for storing the dynamic attributes of Easter Island.


### discretise
Create a discretised representation of Easter Island via triangular cells

using elevation and slope data from Googe Earth Engine
(files Map/elevation_EI.tif and Map/slope_EI.tif)

##### Steps
- load elevation and slope data obtained in a certain latitude/longitude bounding box
- transform pixels to length units (via scaling the original bounding box)
- define interpolation functions of the elevation and slope for continuous points in the bounding box
- create a grid of points (which will be the cell corners) with given resolution (gridpoints_x, gridpoints_y)
- Via the matplotlib.pyplot.triangulation module, define triangular cells on the grid.
- Calculate the midpoints of the resulting triangles
- Mask out the ocean triangles, i.e. those with elevation at the midpoint below 10cm
- Evaluate the elevation and slope at the midpoints of cells on land

### init_trees
distribute trees on cells according to different scenarios
- "uniform" pattern (with equal probability for trees on cells with low enough elevation and slope) or
- "mosaic" pattern (with decreasing probability for trees with the distance to the closest lake/lake_area
to the power of "tree_decrease_lake_distance" and zero probability for cells with too high elevation or slope)
The distance to the closest lake/lake_area corresponds to the inverse water penalty $P_w$
The exponent given by "tree_decrease_lake_distance" allows for determining the degree to
    which the tree probability decreases with the penalty:
    - if exponent==0: prob_trees = equivalent to uniform,
    - if exponent==1: prob_trees = max_lake (area_lake / distance_lake^2)
    - if exponent>1: stronger heterogeneity (faster decrease of probability with area-weighted distance)

### get_agriculture
Assign farming productivity indices to cells
given the classification criteria by Puleston et al. 2017 (Figure 4)

Steps:
- Obtain data from Puleston et al. 2017 (Figure 4)
- Scale data to fit the previously defined grid with length units km
- Evaluate farming productivity indices on the midpoints of cells.
        
