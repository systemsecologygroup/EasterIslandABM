"""
    File name: create_map.py
    Author: Peter Steiglechner
    Date created: 01 December 2020
    Date last modified: 12 April 2021
    Python Version: 3.8
"""

import scipy.spatial.distance
from scipy.interpolate import RectBivariateSpline
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class Map:
    """
    Discretised representation of Easter Island.

    Idea
    ====
    - Obtain an elevation map of a rectangular extract of Easter Island. Each pixel represents the elevation at the
        particular point
        - Here, we use data from Google Earth Engine
        - Define a (coarse) grid of points within this map.
        - Define triangular cells in this grid.
        - Each cell is the smallest unit of our discretised representation of Easter Island.
        - A cell has the following constant properties:
            - elevation
            - slope
            - corresponding geography penalty.
            - midpoint
            - Area
            - farming productivity index
            - number of available well and poorly suited gardens
            - tree carrying capacity (or trees at time of arrival)

        - A cell has additionally dynamic properties:
            - trees
            - smallest area-weighted distance to freshwater lake (depending on droughts)
            - population
            - number of trees cleared
            - number of occupied gardens

        - Additionally a cell can be:
            - on land or ocean
            - the special cell containing Anakena beach (the landing spot of the Rapa Nui),
            - part of a freshwater lake
            - a coastal cell

    Implementation details
    ======================
        Static Variables
        ================
        - triobject : consisting of the matplotlib.triangulation object
            - x : coordinate of points in km
            - y : coordinate of points in km
            - triangles : the indices of the three corner points for each triangular cell
            - mask : denoting those triangles not on the ocean, i.e. where the midpoint has a non-zero elevation
        - el_map : elevation on each cell in m above sea level
        - sl_map : slope of the territorry on each cell in degrees
        - f_pi_c : the farming productivity index (f_pi_well for well suited and f_pi_poor for poorly suited, 0 else)
                for each cell
        - trees_cap : number of trees in each cell before arrival of the first settlers, thus, the cell's tree carrying
                capacity
        - avail_well_gardens : Number of potential well-suited gardens in each cell determined by the area of a cell and
                the size of a garden
        - avail_poor_gardens : Number of potential well-suited gardens in each cell determined by the area of a cell and
                the size of a garden

        Dynamic Variables
        =================
        - water_cells_map : indices of cells containing water
        - penalty_w : Penalty [0,1] for each cell depending on the smallest area-weighted distance to any freshwater
                lake, which depends on whether Rano Raraku is dried out.
        - occupied_gardens : dynamic number of occupied gardens by agents in each cell
        - population_size : population in each cell
        - tree_clearance : number of cleared trees in each cell
        - trees_map : number of available trees in each cell.

    The class needs as input
    ========================
    - an elevation image (with a latitude and longitude bounding box)
    - a slope image (with the same bounding box)
    - Figure 4 of [Puleston2017], giving farming producitivity indices
            (with a longitude and latitude bounding box)

    [Puleston2017]
        Puleston, C. O., Ladefoged, T. N., Haoa, S., Chadwick, O. A., Vitousek, P. M., & Stevenson, C. M. (2017). Rain,
        sun, soil, and sweat: a consideration of population limits on Rapa Nui (Easter Island) before European Contact.
        Frontiers in Ecology and Evolution, 5, 69.
    """

    def __init__(self, m, el_image_file, sl_image_file, puleston2017_image_file, el_bbox, puleston_bbox):
        """
        create the discretised representation of Easter Island and assign values for all variables in all cells
         stored in numpy arrays with one entry per cell.

        Steps
        =====
        - Create a discretised representation of Easter Island via triangular cells
        - calculate the distance matrix of all land cells
        - determine cells that are at the coast and the distance of all land cells to the nearest coast.
        - determine cell of Anakena Beach and the cells within the initial moving radius
        - get water cells for the Lakes Rano Aroi, Kau and Raraku and determine the cells, water penalties and distance
                to water in the periods when Raraku is dried out and when it is not.
        - calculate penalty of elevation and slope and combine them to geography penalty
        - get the farming productivity indices and the amount of available farming gardens in each cell
        - distribute trees on the island as the tree carrying capacity
        - Initialise arrays for storing the dynamic attributes of Easter Island.

        Parameters
        ----------
        m : instance of class Model
            the corresponding model hosting the map
        el_image_file : str
            path and filename of the elevation image
        sl_image_file : str
            path and filename of the slope image
        el_bbox : list of floats
            bounding box of the elevation image: lonmin, latmin, lonmax, latmax
        puleston2017_image_file : str
            path and filename of the farming productivity image by Puleston 2017
        puleston_bbox : (float, float, float, float)
            bounding box of the image by Puleston 2017: lonmin, latmin, lonmax, latmax
        """

        self.el_image_file = el_image_file
        self.sl_image_file = sl_image_file
        self.el_bbox = el_bbox  # lonmin, latmin, lonmax, latmax
        self.puleston2017_image_file = puleston2017_image_file  #
        self.puleston_bbox = puleston_bbox  # lonmin, latmin, lonmax, latmax

        self.m = m  # Model

        # State Variables:
        # Static
        self.triobject = None  # triangulation object
        self.el_map = None  # elevation in each cell
        self.sl_map = None  # slope in each cell
        self.f_pi_c = None
        self.trees_cap = None
        self.avail_well_gardens = None
        self.avail_poor_gardens = None

        # Dynamic:
        self.water_cells_map = None  # The current indices of land cells with freshwater on them
        self.penalty_w = None  # The current water penalty of land cells
        self.occupied_gardens = None
        self.pop_cell = None
        self.tree_clearance = None
        self.trees_map = None

        # Static Helping dependent variables:
        # These variables are only for making implementation easier and faster, but could easily be derived from the
        #   static variables above
        #
        # Grid Related
        self.midpoints = None  # midpoints of triangles
        self.n_triangles_map = None  # nr of triangles 
        self.x_grid = None  # the grid in x direction; could be retrieved from self.triobject
        self.y_grid = None  # the grid in x direction; could be retrieved from self.triobject
        self.inds_map = None  # indices of the land cells in triobject; np.where(np.invert(self.triobject.mask))[0]
        self.n_triangles_map = None  # number of cells on land; len(self.inds_map)
        self.midpoints_map = None  # midpoints on land;  self.midpoints[self.inds_mao]
        self.triangle_area_m2 = None  # Area of the triangle in m^2
        self.area_map_m2 = None  # Area of Easter Island in the discretised state in m2
        self.n_gardens_percell = None  # Number of gardens per cell (rounded down)
        self.coast_triangle_inds = None  # Indices of cells at the coast
        self.dist_to_coast_map = None  # Distance of all land cells to the neares coast
        # Pre-defined cells containing freshwater lakes
        self.water_cells_map_nodrought = None  # The indices of land cells covering ranos aroi, kau and raraku
        self.water_cells_map_drought = None  # The indices of land cells covering ranos aroi and kau
        self.penalty_w_nodrought = None  # Penalties of land cells covering ranos aroi, kau and raraku
        self.penalty_w_drought = None  # Penalties of land cells covering ranos aroi and kau
        self.dist_water_map = None  # Distance of all land cells to closest freshwater lakes rano aroi, kau and raraku
        distmatrix_map = None  # Distance matrix of all land cells;
        # Anakena Beach
        self.anakena_ind = None  # cell of all triangles index of anakena beach
        self.anakena_ind_map = None  # # land cell index of anakena beach
        self.circ_inds_anakena = None  # indices of cells within the initial moving radius of anakena beach
        # Geography penalty
        self.penalty_g = None  # geographic Penalty
        # Matrix of cells within r_T and r_F distances.
        self.circ_inds_trees = None  # square boolean matrix for all land cells: Where distance < r_T of the cell
        self.circ_inds_farming = None  # square boolean matrix for all land cells: Where distance < r_F of the cell

        # === Create the Map ===
        # Calculate the Grid
        self.discretise(m.gridpoints_x, m.gridpoints_y)

        # === Calc Area ===
        self.calculate_triangle_areas()

        # === Distance Matrices ===
        # Calculate the distances between the midpoints of each cell of EI
        distmatrix_map = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(self.midpoints_map)).astype(np.float)

        # === Determine Coast Cells and distances to nearest coast ===
        self.get_coast_cells(distmatrix_map)

        # === Anakena Beach ===
        # Determine the cell belonging to anakena beach, the arrival spot of the first settlers
        anakena_coords = (-27.07327778, -109.32305556)
        self.get_anakena_info(distmatrix_map, anakena_coords)

        # === Calc Geography Penalty ===
        self.penalty_g = np.zeros(self.n_triangles_map, dtype=np.float)
        self.calc_penalty_g()  # calculate the static penalty of geography

        # === Get Freshwater lakes ===
        # Coordinates of Freshwater sources
        raraku = {"midpoint": [-27.121944, -109.2886111], "Radius": 170e-3,
                  "area": np.pi * (170e-3) ** 2}  # Radius in km
        kau = {"midpoint": [-27.186111, -109.4352778], "Radius": 506e-3, "area": np.pi * (506e-3) ** 2}  # Radius in km
        aroi = {"midpoint": [-27.09361111, -109.373888], "Radius": 75e-3, "area": np.pi * (75e-3) ** 2}  # Radius in km
        # calculate which cells have freshwater in two scenarios: Drought and No Drought of Rano Raraku
        self.water_cells_map_nodrought, area_corresp_lake_nodrought = self.setup_freshwater_lakes(distmatrix_map,
                                                                                                  [raraku, kau, aroi])
        self.water_cells_map_drought, area_corresp_lake_drought = self.setup_freshwater_lakes(distmatrix_map,
                                                                                              [kau, aroi])
        # For both scenarios (dorught/nodrought) calculate the penalty for all cells
        self.penalty_w_nodrought, self.dist_water_map = self.calc_penalty_w(distmatrix_map,
                                                                            self.water_cells_map_nodrought,
                                                                            area_corresp_lake_nodrought)
        self.penalty_w_drought, _ = self.calc_penalty_w(distmatrix_map, self.water_cells_map_drought,
                                                        area_corresp_lake_drought)

        # === Calc Resource Access ===
        # for each cell on the map, get all map cells that are within r_t and r_f distance, respectively.
        # if circ_inds_trees[c, c2] == True, then c2 is in r_t distance to c.
        # Agent in c can then harvest trees in c2.
        # An agent in c can loop through the cells with value true in circ_inds_...
        self.circ_inds_trees = np.array(distmatrix_map < self.m.r_t, dtype=bool)
        self.circ_inds_farming = np.array(distmatrix_map < self.m.r_f, dtype=bool)

        # === Agriculture ===
        print("Calculating the farming producitivity index f_pi_c for each cell and the amount of arable gardens")
        self.get_agriculture()

        # === Trees ===
        self.init_trees()

        # === Storage, Initial State: ===
        self.pop_cell = np.zeros_like(self.inds_map, dtype=np.uint64)
        self.tree_clearance = np.zeros_like(self.inds_map, dtype=np.uint64)
        self.occupied_gardens = np.zeros_like(self.inds_map, dtype=np.uint8)
        self.water_cells_map = np.copy(self.water_cells_map_nodrought)
        self.penalty_w = self.penalty_w_nodrought

        return

    def discretise(self, gridpoints_x, gridpoints_y):
        """
        Create a discretised representation of Easter Island via triangular cells

        using elevation and slope data from Googe Earth Engine
        (files Map/elevation_EI.tif and Map/slope_EI.tif)

        Steps
        =====
        - load elevation and slope data obtained in a certain latitude/longitude bounding box
        - transform pixels to length units (via scaling the original bounding box)
        - define interpolation functions of the elevation and slope for continuous points in the bounding box
        - create a grid of points (which will be the cell corners) with given resolution (gridpoints_x, gridpoints_y)
        - Via the matplotlib.pyplot.triangulation module, define triangular cells on the grid.
        - Calculate the midpoints of the resulting triangles
        - Mask out the ocean triangles, i.e. those with elevation at the midpoint below 10cm
        - Evaluate the interpolation functions for elevation and slope at the midpoints of cells on land

        Parameters
        ----------
        gridpoints_x : int
            Number of gridpoints in x (longitudinal) direction
        gridpoints_y : int
            Number of gridpoints in y (latitudinal) direction
        """
        # Elevation and Slope data is taken from Google Earth Engine:
        # Bounding Box is from  -27.205 N to -27.0437 N and -109.465 E to -109.2227 E

        # Read in Elevation Data
        # black-white 8-bit integers: 0..255
        el_image = plt.imread(self.el_image_file)
        # Convert data: 500m is the maximum elevation set in Google Earth Engine Data
        el_image = el_image.astype(float) * 500 / 255

        # Read in Slope Data
        sl_image = plt.imread(self.sl_image_file)
        # Convert data: 30 degree is the maximum slope set in Google Earth Engine Data
        sl_image = sl_image.astype(float) * 30 / 255

        # === Transform pixel elevation image to km ===
        #
        # Bounding Box in degrees of the images
        lonmin, latmin, lonmax, latmax = self.el_bbox

        pixel_dim = el_image.shape

        # get delta latitude per pixel
        d_gradlat_per_pixel = abs(latmax - latmin) / pixel_dim[0]  # [degrees lat per pixel]
        # get delta km per pixel in y-direction
        # according to wikipedia 1deg Latitude = 111.32km
        d_km_pix_y = 111.320 * d_gradlat_per_pixel  # [km/lat_deg * lat_deg/pixel = km/pixel]

        # get delta longitude per pixel
        d_gradlon_per_pixel = abs(lonmax - lonmin) / pixel_dim[1]  # [degrees lon per pixel]
        # get delta km per pixel in x-direction
        # 1deg longitude = 111.32km * cos(latitude)
        cos_lat = abs(np.cos((latmax + latmin) * 0.5 * np.pi / 180))
        d_km_pix_x = 111.320 * cos_lat * d_gradlon_per_pixel  # [km/pixel]]

        # x and y grid [in km units] of pixels for the elevation/slope image
        x_grid_image = np.linspace(
            0 * d_km_pix_x,
            pixel_dim[1] * d_km_pix_x,
            pixel_dim[1],  # nr of pixels
            endpoint=False
        )
        y_grid_image = np.linspace(
            0 * d_km_pix_y,
            pixel_dim[0] * d_km_pix_y,
            pixel_dim[0],  # nr of pixels
            endpoint=False
        )
        # Elevation image bounding box in km
        bbox = [0, x_grid_image[-1], 0, y_grid_image[-1]]

        # interpolation functions on the elevation pixel image
        f_el = RectBivariateSpline(x_grid_image, y_grid_image, el_image.T, bbox=bbox, kx=3, ky=3)
        f_sl = RectBivariateSpline(x_grid_image, y_grid_image, sl_image.T, bbox=bbox, kx=3, ky=3)

        # === Create Grid of cells ===
        # Define grid of cell corner points.
        # The bounds are the same as for the elevation image, but the number of points is set manually by parameters:
        # gridpoints_x and gridpoints_y
        self.x_grid = np.linspace(0 * d_km_pix_x,
                                  pixel_dim[1] * d_km_pix_x,
                                  gridpoints_x,  # set nr of points
                                  endpoint=False)
        self.y_grid = np.linspace(0 * d_km_pix_y,
                                  pixel_dim[0] * d_km_pix_y,
                                  gridpoints_y,  # set nr of points
                                  endpoint=False)
        # all gridpoints of the mesh
        gridpoints = np.array([[[x, y] for y in self.y_grid] for x in self.x_grid])  # TODO use meshgrid

        # === Triangulate ===
        self.triobject = mpl.tri.Triangulation(gridpoints[:, :, 0].flatten(), gridpoints[:, :, 1].flatten())

        # === Find the land cells ===
        # get the midpoints of the triangular cells for each cell
        self.midpoints = np.array([np.sum([[self.triobject.x[d], self.triobject.y[d]] for d in k], axis=0) / 3.0
                                   for k in self.triobject.triangles])

        # Set mask: False if cell on land (el>0.1m), True if not.
        mask = [False if f_el(d[0], d[1])[0][0] > 0.1 else True for d in self.midpoints]
        self.triobject.set_mask(mask)

        self.inds_map = np.where(np.invert(self.triobject.mask))[0]
        self.n_triangles_map = len(self.inds_map)
        self.midpoints_map = self.midpoints[self.inds_map]

        # === Get elevation/slope of the midpoints for all cells on the map ===
        el_all = np.array(
            [f_el(self.midpoints[k, 0], self.midpoints[k, 1])[0][0] for k in range(len(self.midpoints))])
        self.el_map = np.array([el_all[k] for k, m in enumerate(self.triobject.mask) if not m])

        sl_all = np.array(
            [f_sl(self.midpoints[k, 0], self.midpoints[k, 1])[0][0] for k in range(len(self.midpoints))])
        self.sl_map = np.array([sl_all[k] for k, m in enumerate(self.triobject.mask) if not m])

        return

    def get_coast_cells(self, distmatrix_map):
        """
        determine cells that are at the coast and the distance of all land cells to the nearest coast.
        """
        # How many ocean neighbours (i.e. masked out triangles and thus with index -1) does a cell have.
        nr_ocean_nbs = np.sum(self.triobject.neighbors == -1, axis=1)
        # coast cell if at least one but not all neighbour cells are ocean cells
        self.coast_triangle_inds = np.where((nr_ocean_nbs > 0) * (nr_ocean_nbs < 3))[0]
        # calculate distance of each cell on the map to the coast cells.
        coast_triangle_land_cells = [np.where(self.inds_map == i)[0][0] for i in self.coast_triangle_inds]
        self.dist_to_coast_map = np.min(distmatrix_map[coast_triangle_land_cells, :], axis=0)
        return

    def get_anakena_info(self, distmatrix_map, anakenacoords):
        """
        determine cell of Anakena Beach (index of land cells and index of total cells) and the cell indices that fall
            within the initial moving radius of Anakena Beach
        """
        anakena_coords_km = self.from_latlon_tokm(anakenacoords)
        self.anakena_ind = self.triobject.get_trifinder()(anakena_coords_km[0], anakena_coords_km[1])
        if self.anakena_ind == -1:
            print("Error: Anakena Beach Coordinates are on a cell on the ocean. ", anakena_coords_km)
        self.anakena_ind_map = np.where(self.inds_map == self.anakena_ind)[0][0]
        self.circ_inds_anakena = np.where(distmatrix_map[:, self.anakena_ind_map] < self.m.moving_radius_arrival)[0]
        return

    def from_latlon_tokm(self, point):
        """
        calculate the corresponding cell of a point given in lat and lon coordinates.

        Parameters
        ----------
        point : (float, float)
            (latitude, longitude)

        Returns
        -------
        point_km : (float, float)
            corresponding point in km units
        """
        # point in [-27.bla, -109,...]
        lat, lon = point
        lonmin, latmin, lonmax, latmax = self.el_bbox
        # grids of the corners in lat/lon
        grid_y_lat = np.linspace(latmin, latmax, num=len(self.y_grid))
        grid_x_lon = np.linspace(lonmin, lonmax, num=len(self.x_grid))
        # point in x and y coordinates:
        # Note: y is defined from top to bottom, the minus
        cell_y = self.y_grid[-np.where(grid_y_lat > lat)[0][0]]
        cell_x = self.x_grid[np.where(grid_x_lon > lon)[0][0]]
        point_km = [cell_x, cell_y]
        return point_km

    def calculate_triangle_areas(self):
        """
        calculate the area of a triangle, the whole land mass and the number of gardens per cell
        """
        # get three corner points (each with x and y coord) of one triangle (with index 100)
        a, b, c = [np.array([self.triobject.x[k], self.triobject.y[k]]) for k in self.triobject.triangles[100]]
        # Area of the triangle = 1/2 * |AC x AB|_z   (x = cross product)
        self.triangle_area_m2 = 1e6 * abs(0.5 * ((c - a)[0] * (b - a)[1] - (c - a)[1] * (b - a)[0]))  # in m^2
        # Area of Easter Island in the discretised state
        self.area_map_m2 = self.triangle_area_m2 * self.n_triangles_map
        # Number of gardens per cell (rounded down)
        self.n_gardens_percell = int(self.triangle_area_m2 / self.m.garden_area_m2)
        print("Area of triangles in m^2: {}; Area of discretised EI: {}; Nr of gardens per cell: {}".format(
            self.triangle_area_m2, self.area_map_m2, self.n_gardens_percell))
        return

    def calc_penalty_g(self):
        """
        calculate penalty of elevation and slope and combine them to geography penalty P_g
        """
        penalty_el = self.m.P_cat(self.el_map, "el")
        penalty_sl = self.m.P_cat(self.sl_map, "sl")
        self.penalty_g = 0.5 * (penalty_sl + penalty_el)
        # For cells with freshwater, set geography penalty to infinite
        self.penalty_g[self.water_cells_map] = 1e6
        return

    def setup_freshwater_lakes(self, distmatrix_map, lakes):
        """
        determine which cells belong to the freshwater lakes specified by parameter "lakes"

        Parameters
        ----------
        distmatrix_map : np.array([self.n_triangles_map, self.n_triangles_map])
            Distance matrix
        lakes : list of dicts
            list of lake dicts with keywords midpoint, radius, area
        Returns
        -------
        water_cells_map : array of int
            Indices of the map cells within the freshwater lakes.
        area_corresp_lake: array of int
            area of the corresponding lake for each cell of water_cells_map
        """
        area_corresp_lake = []
        water_cells_map = []
        for n, x in enumerate(lakes):
            m = self.from_latlon_tokm(x["midpoint"])
            r = x["Radius"]
            # get triangle of the midpoint of the lake
            triangle = self.triobject.get_trifinder()(m[0], m[1])
            triangle_map = np.where(triangle == self.inds_map)[0][0]
            # get all triangles with midpoints within the lake radius distance
            inds_within_lake = np.where((distmatrix_map[:, triangle_map] < r))[0]
            # Create a list of triangle indices which incorporate the lakes
            if len(inds_within_lake) == 0:
                inds_within_lake = [t]
            water_cells_map.extend(inds_within_lake)  # EI_range
            # Store the area for the corresponding lake for penalty calculation
            area_corresp_lake.extend([x["area"] for _ in range(len(inds_within_lake))])
        water_cells_map = np.array(water_cells_map)
        return water_cells_map, area_corresp_lake

    def calc_penalty_w(self, distmatrix_map, water_cells_map, area_corresp_lake):
        """
        Calculate water penalty P_w

        Using evaluation variable:
        w = min_{lake l} \ d_{l}^2 / A_l,
        where d is the distance to the lake l and A is the area of that lake
        and the given thresholds w01 and w99.

        Parameters
        ----------
        distmatrix_map : np.array([self.n_triangles_map, self.n_triangles_map])
            Distance matrix
        water_cells_map : array of ints
            Indices of the map cells within the freshwater lakes.
        area_corresp_lake : list of floats

        Returns
        -------
        penalty_w : array of floats
            water penalty for each cell
        min_distance_to_water : array of floats
            min distance to water
        """
        # Distance from all cells to all cells containing freshwater
        distances_to_water = distmatrix_map[water_cells_map, :]
        # Weigh distance by the size of the lake
        # Note following line: Casting to divide each row seperately
        weighted_squ_distance_to_water = distances_to_water ** 2 / np.array(area_corresp_lake)[:, None]
        # Take the minimum of the weighted distances to any of the cells containing water
        w_evaluation = np.min(weighted_squ_distance_to_water, axis=0)
        # k_w = self.m(self.m.w01, self.m.w99)
        # Calculate penalty from that
        penalty_w = self.m.P_cat(w_evaluation, "w")

        # print("Water Penalties Mean: ", "%.4f" % (np.mean(P_W)))
        return penalty_w, np.min(distances_to_water, axis=0).clip(1e-10)

    def check_drought(self, t):
        """
        assign freshwater lake cells and the water penalty values for all cells depending on whether Rano Raraku is
            dried out or not

        Note: Parameter self.m.droughts_rano_raraku lists droughts. Each entry is a list of start and end year of the
            drought at Rano Raraku

        Parameters
        ----------
        t : int
            current time
        """
        for drought in self.m.droughts_rano_raraku:
            if t == drought[0]:
                print("beginning drought in Raraku, t=", t)
                self.penalty_w = self.penalty_w_drought
                self.water_cells_map = self.water_cells_map_drought
                # need to calculate geography penalty again because rano raraku cells are dry now.
                self.calc_penalty_g()
            if t == drought[1]:
                # end drought:
                print("ending drought in Raraku, t=", t)
                self.penalty_w = self.penalty_w_nodrought
                self.water_cells_map = self.water_cells_map_nodrought
                self.calc_penalty_g()
        return

    def init_trees(self):
        """
        distribute trees on various cells according to different scenarios

        Defined Scenarios
        =================
        - "uniform" pattern (with equal probability for trees on cells with low enough elevation and slope) or
        - "mosaic" pattern (with decreasing probability for trees with the distance to the closest lake/lake_area
            to the power of "tree_decrease_lake_distance" and 0 probability for cells with too high elevation or slope)
            The distance to the closest lake/lake_area corresponds to the inverse water penalty P_w
            The exponent given by "tree_decrease_lake_distance" allows for determining the degree to
                which the tree probability decreases with the penalty:
                - if exponent==0: prob_trees = equivalent to uniform,
                - if exponent==1: prob_trees = max_lake (area_lake / distance_lake^2)
                - if exponent>1: stronger heterogeneity (faster decrease of probability with area-weighted distance)

        """
        print("Initialising {} Trees on cells with elevation smaller than {}, slope smaller than {} ".format(
            self.m.n_trees_arrival, self.m.map_tree_pattern_condition["max_el"],
            self.m.map_tree_pattern_condition["max_sl"]) +
              "and decreasing density with the min area-weighted distance to a freshwater lake with exponent {}".format(
                  self.m.map_tree_pattern_condition["tree_decrease_lake_distance"])
              if self.m.map_tree_pattern_condition["tree_decrease_lake_distance"] > 0 else "uniformely distributed")

        # Trees are restricted to cells with elevation below a certain elevation and a certain slope
        max_el, max_sl = (self.m.map_tree_pattern_condition["max_el"], self.m.map_tree_pattern_condition["max_sl"])
        # Calculate the probability for trees to be in each cell
        prob_trees = np.ones_like(self.inds_map, dtype=np.float)
        # On water cells: Probability 0
        prob_trees[self.water_cells_map_drought] = 0
        # if tree_decrease_lake_distance == 0 : "Uniform Pattern"
        # if tree_decrease_lake_distance == 1 : "Mosaic Pattern"
        # penalty_w_nodrought = min_lake ( distance_lake^2 / area_lake )
        exponent = self.m.map_tree_pattern_condition["tree_decrease_lake_distance"]
        if not exponent == 0:
            prob_trees = 1 / self.penalty_w_nodrought ** exponent
        # Set prob to zero if slope/elevation are too high
        prob_trees[np.where(self.el_map > max_el)] = 0.
        prob_trees[np.where(self.sl_map > max_sl)] = 0.
        prob_trees *= 1 / np.sum(prob_trees)

        # === Init trees ===
        # distribute n_trees_arrival on the cells with prob_trees[c] for each cell
        inds_all_trees = np.random.choice(range(len(self.inds_map)),
                                          size=int(self.m.n_trees_arrival),
                                          p=prob_trees
                                          )
        # get how often each value (with index in values) was chosen for a tree
        values, counts = np.unique(inds_all_trees, return_counts=True)
        self.trees_map = np.zeros_like(self.inds_map, dtype=np.int32)
        self.trees_map[values] = counts
        self.trees_cap = np.copy(self.trees_map)

        # ========== Check: ===========
        n_cells_with_trees = np.sum(self.trees_cap > 0)
        print("Cells with trees on them: {} (fraction {})".format(n_cells_with_trees,
                                                                  n_cells_with_trees / self.n_triangles_map))
        print("Total trees: {}, Mean trees on cells: {}, Std: {} ".format(
            np.sum(self.trees_map), np.mean(self.trees_map), np.std(self.trees_map)))
        return

    def get_agriculture(self):
        """
        Assign farming productivity indices to cells given the classification criteria by Puleston et al. (2017)
            (Figure 4)

        Steps
        =====
        - Obtain data from Puleston et al. (2017) (Figure 4)
        - Scale data to fit the previously defined grid with length units km
        - Evaluate farming productivity indices on the midpoints of cells from the colors of the figure
        """
        # === Read in data ===
        pulestonMap = plt.imread(self.puleston2017_image_file) * 1 / 255

        # The following colors encode our classification into farming productivity:
        #   well-suited sites = Dark Green (rgb = ca. 0.22, 0.655, 0)
        #   poorly suited sites = bright green (rgb = ca. 0.333, 1, 0),
        #   not suitable = red (rgb = ca. 1 0 0)
        #
        # Array of bools where pixels of the image are well-suited
        data_wellSites = (pulestonMap[:, :, 0] < 0.9) * (pulestonMap[:, :, 1] > 0.5) * (pulestonMap[:, :, 1] < 0.8) * (
                pulestonMap[:, :, 2] < 0.01)
        # Array of bools where pixels of the image are poorly suited.
        # Note: ~ is inversion. Hence a well-suited pixel can not also be poorly suited
        data_poorSites = (pulestonMap[:, :, 0] < 0.9) * (pulestonMap[:, :, 1] >= 0.8) * (
                    pulestonMap[:, :, 2] < 0.01) * (
                             ~data_wellSites)

        # === Transform into our grid ===
        # By comparing the pictures of the google elevation data and Puleston's farming productivity, define the
        # latitude and longitude borders of Puleston's picture
        # This can be adjusted until the data fits roughly.
        dlonmin, dlatmin, dlonmax, dlatmax = [p - b for p, b in zip(self.puleston_bbox, self.el_bbox)]
        lonmin, latmin, lonmax, latmax = self.puleston_bbox

        # transform pixel to km
        # Same as above for the elevation/slope image
        pixel_dim = pulestonMap.shape

        # get delta latitude per pixel
        d_gradlat_per_pixel = abs(latmax - latmin) / pixel_dim[0]  # [degrees lat per pixel]
        # get delta km per pixel in y-direction
        # according to wikipedia 1deg Latitude = 111.32km
        d_km_pix_y = 111.320 * d_gradlat_per_pixel  # [km/lat_deg * lat_deg/pixel = km/pixel]

        # get delta longitude per pixel
        d_gradlon_per_pixel = abs(lonmax - lonmin) / pixel_dim[1]  # [degrees lon per pixel]
        # get delta km per pixel in x-direction
        # 1deg longitude = 111.32km * cos(latitude)
        cos_lat = abs(np.cos((latmax + latmin) * 0.5 * np.pi / 180))
        d_km_pix_x = 111.320 * cos_lat * d_gradlon_per_pixel  # [km/pixel]]

        # The shift of the corner in the images:
        x_lower = 0 + 111.320 * cos_lat * dlonmin
        y_lower = 0 - 111.320 * dlatmax
        # The minus neede here because we define the map bottom up

        # x and y grid in km for the picture of Puleston 2017
        x_grid_puleston = np.linspace(
            x_lower + 0 * d_km_pix_x,
            x_lower + pixel_dim[1] * d_km_pix_x,
            pixel_dim[1],
            endpoint=False
        )
        y_grid_puleston = np.linspace(
            y_lower + 0 * d_km_pix_y,
            y_lower + pixel_dim[0] * d_km_pix_y,
            pixel_dim[0],
            endpoint=False)
        bbox_puleston = [0, x_grid_puleston[-1], 0, y_grid_puleston[-1]]

        # Interpolation function on Puleston 2017's pixel grid of Easter Island.
        # returns large value if well-suited (or poorly) and zero else
        f_well = RectBivariateSpline(x_grid_puleston, y_grid_puleston, data_wellSites.T, bbox=bbox_puleston, kx=3, ky=3)
        f_poor = RectBivariateSpline(x_grid_puleston, y_grid_puleston, data_poorSites.T, bbox=bbox_puleston, kx=3, ky=3)

        # Get values at midpoints of Easter Island. for both poor and well-suited data.
        well_allowed_map = np.array([f_well(m[0], m[1])[0][0] for m in self.midpoints_map]) >= 0.01
        # Cells that are already well-suited are additionally excluded from being poorly suited.
        poor_allowed_map = (~(well_allowed_map > 0)) * (
                    np.array([f_poor(m[0], m[1])[0][0] for m in self.midpoints_map]) >= 0.01)

        # Indices of well-suited and poorly suited cells
        inds_well = np.where(well_allowed_map)[0]
        inds_poor = np.where(poor_allowed_map)[0]

        # === Assign Farming Productivity Indices to poor and well cells ===
        self.f_pi_c = np.zeros_like(self.inds_map, dtype=np.float)
        self.f_pi_c[inds_well] = self.m.f_pi_well
        self.f_pi_c[inds_poor] = self.m.f_pi_poor

        self.avail_well_gardens = np.zeros_like(well_allowed_map, dtype=np.uint8)
        self.avail_well_gardens[inds_well] = self.n_gardens_percell
        self.avail_poor_gardens = np.zeros_like(poor_allowed_map, dtype=np.uint8)
        self.avail_poor_gardens[inds_poor] = self.n_gardens_percell
        # sum_well_suited = np.sum(self.well_suited_cells)
        # sum_poorly_suited = np.sum(self.poorly_suited_cells)

        # ===== Check =======
        area_well_suited_m2 = len(inds_well) * self.n_gardens_percell * self.m.garden_area_m2
        area_poor_suited_m2 = len(inds_poor) * self.n_gardens_percell * self.m.garden_area_m2
        print("Well Suited Sites: ", len(inds_well) * self.n_gardens_percell, " on ", len(inds_well), " Cells")
        print("Poor Suited Sites: ", len(inds_poor) * self.n_gardens_percell, " on ", len(inds_poor), " Cells")
        print("Well suited Area: ", area_well_suited_m2, " or as Fraction: ",
              area_well_suited_m2 / (self.n_triangles_map * self.triangle_area_m2))
        print("Poorly suited Area: ", area_poor_suited_m2, " or as Fraction: ",
              area_poor_suited_m2 / (self.n_triangles_map * self.triangle_area_m2))
        return




if __name__ == "__main__":
    """
    Example:
    Create a representation grid given some parameters
    Plot the farming productivity and initial tree densitiy
    """

    from main import Model
    import importlib
    from plot_functions.plot_InitialMap import *

    # ========== LOAD PARAMETERS ===========
    # Import parameters for sensitivity analysis
    sa_mod = importlib.import_module("params.sa.mosaic_pattern_extreme")
    print("sa.params_sensitivity: ", sa_mod.params_sensitivity)

    # Import parameters for scenario
    scenario_mod = importlib.import_module("params.scenarios.full")
    print("scenarios.params_sensitivity: ", scenario_mod.params_scenario)

    # Import const parameters
    const_file = "default_consts"  # "default_consts"  # "default_consts"  # "single_agent_consts" "alternative_consts"
    consts_mod = importlib.import_module("params.consts." + const_file)
    print("const file", const_file)

    # Seed
    seed = 1
    consts_mod.params_const["gridpoints_x"] = 75
    consts_mod.params_const["gridpoints_y"] = 50

    # === RUN ===
    m = Model("Map/", int(seed), consts_mod.params_const, sa_mod.params_sensitivity, scenario_mod.params_scenario)

    # === PLOT ====
    # Plot Map for Farming Productivity Index
    plot_map(m.map, m.map.f_pi_c * 100, r"Agric. Prod. $a_{\rm p}(c)$ [%]", cmap_fp, 0.01, 100, "F_P")

    # Plot Map for Trees.
    plot_map(m.map, m.map.trees_map * 1 / 100, r"Trees $T(c, t_{\rm 0})$ [1000]", cmap_trees, 0, 200, "T_mosaic_extreme")
