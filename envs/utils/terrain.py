import mujoco
import numpy as np
import scipy.ndimage


class TerrainGenerator:

    def __init__(self, model, hfield_id, config):
        self.model = model
        self.hfield_id = hfield_id
        self.tcfg = config.terrain

        self.nrow, self.ncol = model.hfield_nrow[hfield_id], model.hfield_ncol[hfield_id]
        self.total_length_x, self.total_width_y = model.hfield_size[hfield_id][0:2] * 2.0
        self.pixel_width_x = self.total_length_x / self.ncol
        self.pixel_width_y = self.total_width_y / self.nrow
        if not getattr(config, 'test_mode', False):
            self.cell_length_x = self.total_length_x / self.tcfg.num_levels
            self.cell_width_y  = self.total_width_y / len(self.tcfg.terrain_types)
        else:
            self.cell_length_x = self.total_length_x / self.tcfg.num_lane_cells
            self.cell_width_y  = self.total_width_y
        self.height_data = np.full((self.ncol, self.nrow), self.tcfg.base_height)

    def get_spawn_location(self, row, col):
        start_x = col * self.cell_length_x - self.total_length_x / 2
        margin_x = self.cell_length_x * self.tcfg.feature_margin
        safe_offset = max(margin_x / 2.0, margin_x - 0.5)
        world_x = start_x + safe_offset
        world_y = row * self.cell_width_y - self.total_width_y / 2 + self.cell_width_y / 2
        return (world_x, world_y)

    def _add_rough(self, sub_grid, difficulty):
        nx, ny = sub_grid.shape
        max_roughness = self.tcfg.roughness * difficulty

        if max_roughness == 0.0:
            return

        bx, by = max(1, int(nx * self.tcfg.noise_scale)), max(1, int(ny * self.tcfg.noise_scale))
        small_grid = np.random.uniform(low=-max_roughness, high=max_roughness, size=(bx, by))
        smooth_grid = scipy.ndimage.zoom(small_grid, (nx/bx, ny/by), order=3)

        margin_px = int(nx * self.tcfg.feature_margin)
        smooth_grid[:margin_px, :] = 0.0
        smooth_grid[-int(margin_px / 2):, :] = 0.0
            
        sub_grid += smooth_grid[:nx, :ny]

    def _get_dist_mask(self, shape, faces=1):
        """
        Returns a 2D array where each pixel's value is its distance (in pixels)
        from the relevant edges
        """
        nx, ny = shape
        x_indices, y_indices = np.arange(nx), np.arange(ny)
        X, Y = np.meshgrid(x_indices, y_indices, indexing='ij')

        dist_front = X
        dist_back  = (nx - 1) - X
        dist_left  = Y
        dist_right = (ny - 1) - Y

        if faces == 1:
            return dist_front
        elif faces == 2:
            return np.minimum(dist_front, dist_back)
        elif faces == 4:
            dist_x = np.minimum(dist_front, dist_back)
            dist_y = np.minimum(dist_left, dist_right)
            aspect_ratio = nx / ny
            return np.minimum(dist_x, dist_y * aspect_ratio)

    def _add_staircase(self, sub_grid, difficulty, faces=1, is_up=True):
        nx, ny = sub_grid.shape
        step_height = self.tcfg.step_height * difficulty

        step_depth_px = max(1, int(self.tcfg.step_width / self.pixel_width_x))
        margin_px = int(nx * self.tcfg.feature_margin)

        dist_map = self._get_dist_mask((nx, ny), faces)

        step_index = (dist_map - margin_px) // step_depth_px
        step_index = np.clip(step_index, 0, None)

        direction = 1 if is_up else -1
        sub_grid += (step_index * step_height * direction)
        sub_grid[nx - int(margin_px / 2):] = self.tcfg.base_height

    def _add_hill_pit(self, sub_grid, difficulty, faces=1, is_up=True):
        nx, ny = sub_grid.shape
        max_height = self.tcfg.hill_height * difficulty
        margin_px = int(nx * self.tcfg.feature_margin)

        if faces in [1, 2]:
            dist_map = self._get_dist_mask((nx, ny), faces)

            dist_map = np.clip(dist_map, 0, nx - margin_px)
            dist_map = np.clip(dist_map - margin_px, 0, None)

            peak_dist = np.max(dist_map)
            norm_dist = dist_map / peak_dist if peak_dist > 0 else dist_map

            smooth_curve = (1.0 - np.cos(norm_dist * np.pi)) / 2.0
        elif faces == 4:
            # To completely smooth out the diagonal "pyramid creases" of a face=4 map,
            # we create a true 2D Cosine Bell by multiplying the X and Y curves.
            x_indices, y_indices = np.arange(nx), np.arange(ny)
            X, Y = np.meshgrid(x_indices, y_indices, indexing='ij')

            dist_x = np.minimum(X, (nx - 1) - X)
            dist_y = np.minimum(Y, (ny - 1) - Y)
            dist_x = np.clip(dist_x - margin_px, 0, None)
            dist_y = np.clip(dist_y - margin_px, 0, None)

            norm_x = dist_x / np.max(dist_x) if np.max(dist_x) > 0 else dist_x
            norm_y = dist_y / np.max(dist_y) if np.max(dist_y) > 0 else dist_y

            curve_x = (1.0 - np.cos(norm_x * np.pi)) / 2.0
            curve_y = (1.0 - np.cos(norm_y * np.pi)) / 2.0

            # Blending them together creates a perfectly round, natural dome
            smooth_curve = curve_x * curve_y

        direction = 1 if is_up else -1
        sub_grid += (smooth_curve * max_height * direction)

    def _build_cell(self, sub_grid, terrain_type, difficulty):
        terrain_map = {
            'stairs_up':   (self._add_staircase, 1, True),
            'stairs_down': (self._add_staircase, 1, False),
            'stair_ridge': (self._add_staircase, 2, True),  # steps up, then steps down
            'stair_trench':(self._add_staircase, 2, False), # steps down, then steps up
            'pyramid':     (self._add_staircase, 4, True),  # steps up from all 4 sides
            'inv_pyramid': (self._add_staircase, 4, False), # stepped square pit
            'ramp_up':     (self._add_hill_pit,  1, True),
            'ramp_down':   (self._add_hill_pit,  1, False),
            'hill':        (self._add_hill_pit,  2, True),  # smooth ridge
            'trench':      (self._add_hill_pit,  2, False), # smooth ditch
            'mound':       (self._add_hill_pit,  4, True),  # smooth round dome
            'bowl':        (self._add_hill_pit,  4, False)  # smooth round crater
        }

        if terrain_type == 'flat':
            pass
        elif terrain_type == 'rough':
            self._add_rough(sub_grid, difficulty)
        elif terrain_type in terrain_map:
            func, faces, is_up = terrain_map[terrain_type]
            func(sub_grid, difficulty, faces=faces, is_up=is_up)
        else:
            raise ValueError(f"Unknown terrain type: {terrain_type}")

    def generate_arena(self):
        pixels_per_cell_x = int(self.cell_length_x / self.pixel_width_x)
        pixels_per_cell_y = int(self.cell_width_y / self.pixel_width_y)

        for col in range(self.tcfg.num_levels):
            fraction = col / max(1, self.tcfg.num_levels - 1)

            for row in range(len(self.tcfg.terrain_types)):
                terrain_type = self.tcfg.terrain_types[row]
                start_x = col * pixels_per_cell_x
                end_x = start_x + pixels_per_cell_x
                start_y = row * pixels_per_cell_y
                end_y = start_y + pixels_per_cell_y

                sub_grid = self.height_data[start_x:end_x, start_y:end_y]
                self._build_cell(sub_grid, terrain_type, fraction)

    def generate_lane(self):
        num_cells = self.tcfg.num_lane_cells
           
        terrain_types = getattr(self.tcfg, 'lane_terrain_types', None)
        difficulties = getattr(self.tcfg, 'lane_difficulties', None)

        if terrain_types is None:
            type_list = [np.random.choice(self.tcfg.terrain_types) for _ in range(num_cells)]
        elif isinstance(terrain_types, str):
            type_list = [terrain_types] * num_cells
        elif isinstance(terrain_types, list):
            type_list = [terrain_types[i % len(terrain_types)] for i in range(num_cells)]

        if difficulties is None:
            diff_list = [np.random.uniform(0.0, 1.0) for _ in range(num_cells)]
        elif isinstance(difficulties, (int, float)):
            diff_list = [float(difficulties)] * num_cells
        elif isinstance(difficulties, list):
            diff_list = [difficulties[i % len(difficulties)] for i in range(num_cells)]

        pixels_per_cell_x = int(self.cell_length_x / self.pixel_width_x)
        pixels_per_cell_y = int(self.cell_width_y / self.pixel_width_y)

        for col in range(num_cells):
            terrain_type = type_list[col]
            fraction = diff_list[col]

            start_x = col * pixels_per_cell_x
            end_x = start_x + pixels_per_cell_x

            sub_grid = self.height_data[start_x:end_x, :]
            self._build_cell(sub_grid, terrain_type, fraction)

    def _apply_to_mujoco(self):
        if not self.tcfg.enabled or self.hfield_id == -1:
            return

        elevation_z = self.model.hfield_size[self.hfield_id][2]
        normalized_data = np.clip(self.height_data / elevation_z, 0.0, 1.0)
        flat_data = normalized_data.T.flatten()
        self.model.hfield_data[:] = flat_data

    def get_height_at(self, x: float, y: float) -> float:
        if self.hfield_id == -1:
            return 0.0

        radius_x, radius_y, elevation_z, _ = self.model.hfield_size[self.hfield_id]

        geom_pos = [0, 0, 0]
        for i in range(self.model.ngeom):
            if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_HFIELD:
                geom_pos = self.model.geom_pos[i]
                break

        # Map world (x, y) to local hfield indices [0, 1]
        local_x = (x - geom_pos[0] + radius_x) / (2.0 * radius_x)
        local_y = (y - geom_pos[1] + radius_y) / (2.0 * radius_y)
        local_x, local_y = np.clip(local_x, 0, 1), np.clip(local_y, 0, 1)

        # Convert to pixel indices
        ix = int(local_x * (self.ncol - 1))
        iy = int(local_y * (self.nrow - 1))

        # Retrieve data (MuJoCo buffer is row-major Y, then X)
        height_raw = self.model.hfield_data[iy * self.ncol + ix]

        # Scale back to meters
        return height_raw * elevation_z + geom_pos[2]
