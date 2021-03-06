import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance
from scipy.interpolate import RectBivariateSpline
import matplotlib as mpl


class Map:
    """
    A brief summary of its purpose and behavior
    Any public methods, along with a brief description
    Any class properties (attributes)
    Anything related to the interface for subclassers, if the class is intended to be subclassed


    """
    def __init__(self, m):
        """

    A brief description of what the method is and what it’s used for
    Any arguments (both required and optional) that are passed including keyword arguments
    Label any arguments that are considered optional or have a default value
    Any side effects that occur when executing the method
    Any exceptions that are raised
    Any restrictions on when the method can be called



        Discretised Map of Easter Island
        Unit of distance is in m or km.
        cells are uniform
        """

        self.m = m
        # need garden_area_m2

        self.triobject = None
        self.midpoints =None
        self.el_map = None
        self.sl_map = None
        self.n_triangles_map = None
        self.box_latlon = None
        self.x_grid = None
        self.y_grid = None
        self.discretise(m.gridpoints_x, m.gridpoints_y)

        # From now on: all variables with "_map" are defined on the cells on Easter island only.
        self.inds_map = np.where(np.invert(self.triobject.mask))[0]
        self.n_triangles_map = len(self.inds_map)
        self.midpoints_map = self.midpoints[self.inds_map]

        # === Calc Area ===
        # get three corner points (each with x and y coord) of one triangle (with index 100)
        a, b, c = [np.array([self.triobject.x[k], self.triobject.y[k]]) for k in self.triobject.triangles[100]]
        # Area of the triangle = 1/2 * |AC x AB|_z   (x = cross product)
        self.triangle_area_m2 = 1e6 * abs(0.5 * ((c-a)[0] * (b - a)[1] - (c-a)[1] * (b-a)[0]))  # in m^2
        # Area of Easter Island in the discretised state
        self.area_map_m2 = self.triangle_area_m2 * self.n_triangles_map
        # Number of gardens per cell (rounded down)
        self.n_gardens_percell = int(self.triangle_area_m2 / self.m.garden_area_m2)
        print("Area of triangles in m^2: {}; Area of discretised EI: {}; Nr of gardens per cell: {}".format(
            self.triangle_area_m2, self.area_map_m2, self.n_gardens_percell))

        # === Distance Matrices ===
        # Calculate the distances between the midpoints of each cell of EI
        distmatrix_map = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(self.midpoints_map)).astype(np.float)

        # === Determine Coast Cells ===
        # Which cells are at the coast.
        # How many ocean neighbours (i.e. masked out triangles and thus with index -1) does a cell have.
        nr_ocean_nbs = np.sum(self.triobject.neighbors == -1, axis=1)
        # coast cell if at least one but not all neighbour cells are ocean cells
        self.coast_triangle_inds = np.where((nr_ocean_nbs > 0) * (nr_ocean_nbs < 3))[0]
        # calculate distance of each cell on the map to the coast cells.
        coast_triangle_inds_map = [np.where(self.inds_map == i)[0][0] for i in self.coast_triangle_inds]
        self.dist_to_coast_map = np.min(distmatrix_map[coast_triangle_inds_map, :], axis=0)

        # === Anakena Beach ===
        # Determine the cell belonging to anakena beach, the arrival spot of the first settlers
        anakena_coords_km = self.from_latlon_tokm((-27.07327778, -109.32305556))
        self.anakena_ind = self.triobject.get_trifinder()(anakena_coords_km[0], anakena_coords_km[1])
        if self.anakena_ind == -1:
            print("Error: Anakena Beach Coordinates Wrong: ", anakena_coords_km)
        self.anakena_ind_map = np.where(self.inds_map == self.anakena_ind)[0][0]
        self.circ_inds_anakena = np.where(distmatrix_map[:, self.anakena_ind_map] < self.m.moving_radius_arrival)[0]

        # === Get Freshwater lakes ===
        # Coordinates of Freshwater sources
        Raraku = {"midpoint": [-27.121944, -109.2886111], "Radius": 170e-3, "area": np.pi*(170e-3)**2}  # Radius in km
        Kau = {"midpoint": [-27.186111, -109.4352778], "Radius": 506e-3, "area": np.pi*(506e-3)**2}  # Radius in km
        Aroi = {"midpoint": [-27.09361111, -109.373888], "Radius": 75e-3, "area": np.pi*(75e-3)**2}  # Radius in km

        # calculate which cells have freshwater in two scenarios: Drought and No Drought of Rano Raraku
        self.water_cells_map_nodrought, area_corresp_lake_nodrought = self.setup_freshwater_lakes(distmatrix_map, [Raraku, Kau, Aroi])
        self.water_cells_map_drought, area_corresp_lake_drought = self.setup_freshwater_lakes(distmatrix_map, [Kau, Aroi])

        # For both scenarios calculate the penalty for all cells
        self.penalty_w_nodrought, dist_water_map_nodrought = self.calc_penalty_w(distmatrix_map, self.water_cells_map_nodrought, area_corresp_lake_nodrought)
        self.penalty_w_drought, _  = self.calc_penalty_w(distmatrix_map, self.water_cells_map_drought, area_corresp_lake_drought)

        # Current Settings (no drought)
        self.water_cells_map = np.copy(self.water_cells_map_nodrought)
        self.penalty_w = self.penalty_w_nodrought
        self.dist_water_map = dist_water_map_nodrought

        # === Calc Trees ===

        # === Calc Geography Penalty ===
        self.penalty_g = np.zeros(self.n_triangles_map, dtype=np.float)
        self.calc_penalty_g()

        # === Calc Resource Access ===
        # for each cell on the map, get all map cells that are within r_t and r_f distance, respectively.
        # if circ_inds_trees[c, c2] == True, then c2 is in r_t distance to c.
        # Agent in c can then harvest trees in c2.
        # An agent in c can loop through the cells with value true in circ_inds_...
        self.circ_inds_trees = np.array(distmatrix_map<self.m.r_t, dtype=bool)
        self.circ_inds_farming = np.array(distmatrix_map<self.m.r_f, dtype=bool)
        # TODO: CHECK the things below
        # self.circ_inds_anakena = np.where(distmatrix_map[:, self.anakena_ind_map] < self.m.moving_radius_arrival)[0]

        # === Agriculture ===
        self.f_pi_c = None
        self.avail_well_gardens = None
        self.avail_poor_gardens = None
        self.occupied_gardens = np.zeros_like(self.inds_map, dtype=np.uint8)
        self.get_agriculture()

         # === Trees ===
        print("Initialising {} Trees on cells with elevation smaller than {}, slope smaller than {} ".format(
            self.m.n_trees_arrival, self.m.map_tree_pattern_condition["max_el"], self.m.map_tree_pattern_condition["max_sl"])+
              "and decreasing density with the area-weighted distance to the closest freshwater lake with exponent {}".format(
                  self.m.map_tree_pattern_condition["tree_decrease_lake_distance"])
              if self.m.map_tree_pattern_condition["tree_decrease_lake_distance"] > 0 else "uniformely distributed")
        self.trees_cap = None
        self.trees_map = None
        self.init_trees()

        # === Storage ===
        self.pop_cell = np.zeros_like(self.inds_map, dtype=np.uint64)
        self.tree_clearance = np.zeros_like(self.inds_map, dtype=np.uint64)
        return





    def from_latlon_tokm(self, point):
        """
        calculate the corresponding cell of a point given in lat and lon coordinates.
        Parameters
        ----------
        point : (float, float)
            (latitude, longitude)
        """
        # point in [-27.bla, -109,...]
        lat, lon = point
        lonmin, latmin, lonmax, latmax = self.box_latlon
        # grids of the corners in
        grid_y_lat = np.linspace(latmin, latmax, num=len(self.y_grid))
        grid_x_lon = np.linspace(lonmin, lonmax, num=len(self.x_grid))
        # point in x and y coordinates:
        # Note: y is defined from top to bottom, the minus
        cell_y = self.y_grid[-np.where(grid_y_lat > lat)[0][0]]
        # TODO Test if this works:  self.y_grid[np.where(grid_y_lat < lat)[0][0]] or with abs
        cell_x = self.x_grid[np.where(grid_x_lon > lon)[0][0]]
        return [cell_x, cell_y]


    def setup_freshwater_lakes(self, distmatrix_map, lakes):
        """
        determine which cells belong to the freshwater lakes specified by Lakes

        Parameters
        ----------
        distmatrix_map : np.array([self.n_triangles_map, self.n_triangles_map])
            Distance matrix
        lakes : list
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
        calculate water penalty from evaluation variable:
        $ w = min_{\rm lake\ l} \ d_{\rm l}^2 / A_l$
        and logistic function $P_{\rm cat}(x)$ with the given thresholds

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
        weighted_squ_distance_to_water = distances_to_water ** 2 / np.array(area_corresp_lake)[:,None]
        # Take the minimum of the weighted distances to any of the cells containing water
        w_evaluation = np.min(weighted_squ_distance_to_water, axis=0)
        # k_w = self.m(self.m.w01, self.m.w99)
        # Calculate penalty from that
        penalty_w = self.m.P_cat(w_evaluation, "w")

        #print("Water Penalties Mean: ", "%.4f" % (np.mean(P_W)))
        return penalty_w, np.min(distances_to_water, axis=0).clip(1e-10)

    def check_drought(self, t):
        """
        checks whether raraku is currently dried out and selects the corresponding water cells on the map and penalties.
        Parameter self.m.droughts_rano_raraku lists droughts with each a list of start and end year.

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

    def calc_penalty_g(self):
        """
        calculate penalty of elevation and slope and combine them to geography penalty
        """
        penalty_el = self.m.P_cat(self.el_map, "el")
        penalty_sl = self.m.P_cat(self.sl_map, "sl")
        self.penalty_g = 0.5 * (penalty_sl + penalty_el)
        # For cells with freshwater, set geography penalty to infinite
        self.penalty_g[self.water_cells_map] = 1e6
        return

    def discretise(self, gridpoints_x, gridpoints_y):
        """
        Create a discretised representation of Easter Island via triangular cells
        using elevation and slope data from Googe Earth Engine
        (files Map/elevation_EI.tif and Map/slope_EI.tif)

        The steps are:
        - load elevation and slope data obtained in a certain latitude/longitude bounding box
        - transform pixels to length units (via scaling the original bounding box)
        - define interpolation functions of the elevation and slope for continuous points in the bounding box
        - create a grid of points (which will be the cell corners) with given resolution (gridpoints_x, gridpoints_y)
        - Via the matplotlib.pyplot.triangulation module, define triangular cells on the grid.
        - Calculate the midpoints of the resulting triangles
        - Mask out the ocean triangles, i.e. those with elevation at the midpoint below 10cm
        - Evaluate the elevation and slope at the midpoints of cells on land

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
        el_image = plt.imread("Map/elevation_EI.tif")
        # Convert data: 500m is the maximum elevation set in Google Earth Engine Data
        el_image = el_image.astype(float) * 500 / 255

        # Read in Slope Data
        sl_image = plt.imread("Map/slope_EI.tif")
        # Convert data: 30 degree is the maximum slope set in Google Earth Engine Data
        sl_image = sl_image.astype(float) * 30 / 255


        # === Transform pixel elevation image to km ===
        #
        # Bounding Box in degrees of the images
        lonmin, latmin, lonmax, latmax = [-109.465, -27.205, -109.2227, -27.0437]
        self.box_latlon = [lonmin, latmin, lonmax, latmax]

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

        # === Get elevation/slope of the midpoints for all cells on the map ===
        el_all = np.array(
            [f_el(self.midpoints[k, 0], self.midpoints[k, 1])[0][0] for k in range(len(self.midpoints))])
        self.el_map = np.array([el_all[k] for k, m in enumerate(self.triobject.mask) if not m])

        sl_all = np.array(
            [f_sl(self.midpoints[k, 0], self.midpoints[k, 1])[0][0] for k in range(len(self.midpoints))])
        self.sl_map = np.array([sl_all[k] for k, m in enumerate(self.triobject.mask) if not m])
        return

    def init_trees(self):
        """
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

        """
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
        self.trees_map = np.zeros_like(self.inds_map, dtype=np.int16)
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
        Assign farming productivity indices to cells
        given the classification criteria by Puleston et al. 2017 (Figure 4)

        Steps:
        - Obtain data from Puleston et al. 2017 (Figure 4)
        - Scale data to fit the previously defined grid with length units km
        - Evaluate farming productivity indices on the midpoints of cells.
        """
        # === Read in data ===
        pulestonMap = plt.imread("Map/puleston2017_original.jpg") * 1 / 255

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
        data_poorSites = (pulestonMap[:, :, 0] < 0.9) * (pulestonMap[:, :, 1] >= 0.8) * (pulestonMap[:, :, 2] < 0.01) * (
            ~data_wellSites)

        # === Transform into our grid ===
        # By comparing the pictures of the google elevation data and Puleston's farming productivity, define the latitude and longitude borders of Puleston's picture
        # This can be adjusted until the data fits roughly.

        # Bounding Box of Elevation Picture
        lonmin, latmin, lonmax, latmax = [-109.465, -27.205, -109.2227, -27.0437]
        #dlonmin, dlonmax = (0.010944113766048511, 7.906722539229678e-05)
        #dlatmin, dlatmax = (0.003903543647363802, -0.007911646499567706)
        dlonmin, dlonmax = (0.01224029778887305, 0.005111285663338093)
        dlatmin, dlatmax = (0.003903543647363802, -0.008886159128012905)

        dlonmin, dlonmax = (0.013,  0.003)
        dlatmin, dlatmax = (0.007, -0.01)


        lonmin, lonmax = (lonmin + dlonmin, lonmax + dlonmax)
        latmin, latmax = (latmin + dlatmin, latmax + dlatmax)

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
        # TODO check why max not min (i guess because we define the map bottom up)

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
        poor_allowed_map = (~(well_allowed_map > 0)) * (np.array([f_poor(m[0], m[1])[0][0] for m in self.midpoints_map]) >= 0.01)

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
        #sum_well_suited = np.sum(self.well_suited_cells)
        #sum_poorly_suited = np.sum(self.poorly_suited_cells)

        # ===== Check =======
        area_well_suited_m2 = len(inds_well) * self.n_gardens_percell * self.m.garden_area_m2
        area_poor_suited_m2 = len(inds_poor) * self.n_gardens_percell * self.m.garden_area_m2
        print("Well Suited Sites: ", len(inds_well)* self.n_gardens_percell , " on ", len(inds_well), " Cells")
        print("Poor Suited Sites: ", len(inds_poor)* self.n_gardens_percell,  " on ", len(inds_poor), " Cells")
        print("Well suited Area: ", area_well_suited_m2, " or as Fraction: ",
              area_well_suited_m2 / (self.n_triangles_map * self.triangle_area_m2))
        print("Poorly suited Area: ", area_poor_suited_m2, " or as Fraction: ", area_poor_suited_m2 / (self.n_triangles_map * self.triangle_area_m2))
        return




if __name__=="__main__":
    from Amain import Model
    from plot_consts import *
    from params_std import params_const, params_sensitivity, params_scenario
    m = Model("", params_const, params_sensitivity, params_scenario)

    # Plot Map for Farming Productivity Index
    plot_map(m.map, m.map.f_pi_c*100, r"Farm. Prod. $F_{\rm P}(c)$ [%]", cmap_fp, 0.01, 100, "F_P")

    # Plot Map for Trees.
    plot_map(m.map, m.map.trees_map * 1/100, r"Trees $T(c, t_{\rm 0})$ [1000]", cmap_trees, 0, 75, "T")



