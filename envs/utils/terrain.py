import numpy as np
import scipy.ndimage
import mujoco
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Any

matplotlib.use('Agg')

class TerrainGenerator:

    def __init__(self, config: Any) -> None:
        """
        Args:
            config: The env's config class containing a `terrain` subclass
        """
        self.config = config
        self.base_h: float = self.config.terrain.base_height

    def _apply_roughness(
        self,
        grid: np.ndarray,
        np_random: np.random.Generator,
        roughness: Optional[float] = None
    ) -> None:
        """
        Adds smooth, rolling organic noise to a grid using upscaled interpolation

        Args:
            grid: The 2D array to modify
            np_random: NumPy random generator
            roughness: Maximum height of noise
        """
        nx, ny = grid.shape
        roughness = roughness or self.config.terrain.roughness
        scale = self.config.terrain.noise_scale

        bx, by = max(1, int(nx * scale)), max(1, int(ny * scale))
        small_grid = np_random.uniform(low=-roughness/2, high=roughness/2, size=(bx, by))
        smooth_grid = scipy.ndimage.zoom(small_grid, (nx/bx, ny/by), order=3)
        grid += smooth_grid[:nx, :ny]

    def _get_dist_mask(self, grid: np.ndarray, faces: int = 4) -> np.ndarray:
        """
        Calculates a distance-from-edge map used for shaping geometric features

        Args:
            grid: The reference grid for dimensions
            faces: Number of directions to rise from (1=front, 2=ridge, 4=pyramid)
            
        Returns:
            A 2D array of distances from relevant edges
        """
        nx, ny = grid.shape
        r = np.arange(nx); c = np.arange(ny)
        X, Y = np.meshgrid(r, c, indexing='ij')
        d_f, d_b = X, (nx - 1 - X)
        d_l, d_r = Y, (ny - 1 - Y)

        if faces == 1:
            return d_f
        elif faces == 2:
            return np.minimum(d_f, d_b)
        else:
            return np.minimum(np.minimum(d_f, d_b), np.minimum(d_l, d_r))

    def _get_boundary_window(
        self,
        grid: np.ndarray,
        m_per_px: float,
        faces: int = 4
    ) -> np.ndarray:
        """
        Generates a 2D smooth S-curve window to blend features into zone boundaries.

        Args:
            grid: The target grid for dimensions
            m_per_px: Resolution scale (meters per pixel)
            faces: Dimensionality of the window (1/2 for X-only, 4 for X-Y)

        Returns:
            A 2D windowing mask [0, 1]
        """
        nx, ny = grid.shape
        r = np.arange(nx); c = np.arange(ny)
        X, Y = np.meshgrid(r, c, indexing='ij')

        # Calculate distance from edges in pixels
        taper_px = self.config.terrain.taper_distance_m / m_per_px

        # Linear ramp from 0 to 1 over the taper_px distance
        win_x = np.clip(np.minimum(X, nx - 1 - X) / taper_px, 0, 1)
        win_y = np.clip(np.minimum(Y, ny - 1 - Y) / taper_px, 0, 1)

        # Transform to S-Curve (Cosine) for organic transitions
        win_x = 0.5 * (1.0 - np.cos(win_x * np.pi))
        win_y = 0.5 * (1.0 - np.cos(win_y * np.pi))
        return win_x if (faces == 1 or faces == 2) else win_x * win_y

    def _apply_stair_geometry(
        self,
        grid: np.ndarray,
        num_steps: int,
        step_height: float,
        step_depth_px: int,
        faces: int = 4
    ) -> None:
        """
        Applies mathematical staircase steps to a grid

        Args:
            grid: The zone to modify
            num_steps: Number of steps to create
            step_height: Vertical height per step
            step_depth_px: Horizontal depth per step in pixels
            faces: Number of faces for the staircase
        """
        dist = self._get_dist_mask(grid, faces)
        margin = int(grid.shape[0] * self.config.terrain.feature_margin)
        step_index = (dist - margin) // step_depth_px
        step_index = np.clip(step_index, 0, num_steps)
        grid += step_index * step_height

    def _apply_natural_hill(
        self,
        grid: np.ndarray,
        height_offset: float,
        np_random: np.random.Generator,
        m_per_px: float,
        faces: int = 4
    ) -> None:
        """
        Applies an asymmetric Gaussian mound with windowed boundary blending

        Args:
            grid: The zone to modify
            height_offset: Peak height/depth
            np_random: NumPy random generator
            m_per_px: Resolution scale
            faces: Number of faces
        """
        nx, ny = grid.shape
        r = np.arange(nx); c = np.arange(ny)
        X, Y = np.meshgrid(r, c, indexing='ij')

        cx = nx // 2 + np_random.integers(-nx//10, nx//10)
        cy = ny // 2 + np_random.integers(-ny//10, ny//10)

        s_min, s_max = self.config.terrain.hill_sigma_range
        sigma_x = np_random.uniform(s_min, s_max) / m_per_px
        sigma_y = np_random.uniform(s_min, s_max) / m_per_px

        exponent = -((X - cx)**2 / (2 * sigma_x**2) + (Y - cy)**2 / (2 * sigma_y**2))
        gaussian = np.exp(exponent)

        window = self._get_boundary_window(grid, m_per_px, faces)
        texture_amp = self.config.terrain.texture_amplitude
        texture = np_random.uniform(-texture_amp, texture_amp, size=(nx, ny))
        grid += gaussian * window * (1.0 + texture) * height_offset

    def _clear_area(
        self,
        grid: np.ndarray,
        m_per_px: float,
        faces: int = 4
    ) -> None:
        """
        Flattens a specific area to the base height using smooth S-curve blending.

        Args:
            grid: The zone to modify
            m_per_px: Resolution scale
            faces: Shape of the clearing area
        """
        dist = self._get_dist_mask(grid, faces)
        blend_px = self.config.terrain.blend_distance_m / m_per_px

        # S-Curve blending ramp [0, 1]
        flatten_mask = np.clip((dist - 2) / blend_px, 0.0, 1.0)
        flatten_mask = 0.5 * (1.0 - np.cos(flatten_mask * np.pi))

        grid[:] = self.base_h + (grid - self.base_h) * (1.0 - flatten_mask)

    def _smooth_seams(self, grid: np.ndarray, nx_zones: int, ny_zones: int) -> np.ndarray:
        """
        Applies a local blur specifically to the boundary lines between zones
        to make the overall terrain look like one continuous piece.
        """
        nx, ny = grid.shape
        zx, zy = nx // nx_zones, ny // ny_zones

        # Create a blurred version of the whole map
        blurred = scipy.ndimage.gaussian_filter(grid, sigma=3.0)

        # Create a soft mask for the seams
        seam_mask = np.zeros_like(grid)
        seam_width = 8 # Increased width for wider blending

        for i in range(1, nx_zones):
            s_s, s_e = i*zx - seam_width, i*zx + seam_width
            seam_mask[max(0, s_s) : min(nx, s_e), :] = 1.0
        for j in range(1, ny_zones):
            s_s, s_e = j*zy - seam_width, j*zy + seam_width
            seam_mask[:, max(0, s_s) : min(ny, s_e)] = 1.0
  
        # Apply Gaussian filter to the seam_mask itself to make the transition soft
        seam_mask = scipy.ndimage.gaussian_filter(seam_mask, sigma=seam_width/2)

        # Blend the blurred seams back into the original grid using the soft mask
        return (1.0 - seam_mask) * grid + seam_mask * blurred

    def create_staircase(
        self,
        grid: np.ndarray,
        num_steps: int,
        step_height: float,
        step_depth_px: int,
        m_per_px: float,
        faces: int = 4
    ) -> None:
        """Creating staircase with pre-clearing"""
        self._clear_area(grid, m_per_px, faces)
        self._apply_stair_geometry(grid, num_steps, step_height, step_depth_px, faces)

    def create_smooth_mound(
        self,
        grid: np.ndarray,
        height: float,
        np_random: np.random.Generator,
        m_per_px: float,
        faces: int = 4
    ) -> None:
        """Creating a hill or pit with pre-clearing"""
        self._clear_area(grid, m_per_px, faces)
        self._apply_natural_hill(grid, height, np_random, m_per_px, faces)

    def _get_terrain_type(self, np_random: np.random.Generator) -> str:
        """Determines the specific terrain feature to generate based on config"""
        choice = self.config.terrain.terrain_type
        if choice == 'random':
            choice = np_random.choice(self.config.terrain.terrain_types)
        return choice

    def _fill_zone_with_feature(
        self, 
        zone: np.ndarray, 
        choice: str, 
        np_random: np.random.Generator, 
        step_depth_px: int, 
        m_per_px: float, 
        faces: int
    ) -> bool:
        """
        Populates a specific grid area with the chosen terrain feature.

        Args:
            zone: The 2D array slice to modify
            choice: The type of terrain (stairs, hill, etc.)
            np_random: Random generator for offsets/steps
            step_depth_px: Horizontal depth per stair step
            m_per_px: Meters per pixel resolution
            faces: Directions for feature rise

        Returns:
            True if the feature is a staircase (requires a mask update for smoothing).
        """
        s_min, s_max = self.config.terrain.stair_step_range
        n_steps = np_random.integers(s_min, s_max)
        h_min, h_max = self.config.terrain.hill_height_range
        h_offset = np_random.uniform(h_min, h_max)

        if choice == 'flat':
            self._clear_area(zone, m_per_px, faces=faces)
        elif choice == 'rough':
            self._apply_roughness(zone, np_random)
        elif choice == 'stairs_up':
            self.create_staircase(zone, n_steps, self.config.terrain.step_height, step_depth_px, m_per_px, faces)
            return True
        elif choice == 'stairs_down':
            self.create_staircase(zone, n_steps, -self.config.terrain.step_height, step_depth_px, m_per_px, faces)
            return True
        elif choice == 'hill':
            self.create_smooth_mound(zone, h_offset, np_random, m_per_px, faces)
        elif choice == 'pit':
            self.create_smooth_mound(zone, -h_offset, np_random, m_per_px, faces)
        return False

    def generate_grid_terrain(
        self,
        np_random: np.random.Generator,
        nx: int,
        ny: int,
        step_depth_px: int,
        m_per_px: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds a grid of randomized obstacles based on the configuration zones

        Args:
            np_random: NumPy random generator
            nx: Number of pixels in X
            ny: Number of pixels in Y
            step_depth_px: Calculated step depth
            m_per_px: Resolution scale

        Returns:
            Tuple of (full_grid, staircase_mask).
        """
        nx_zones = self.config.terrain.nx_zones
        ny_zones = self.config.terrain.ny_zones
        spawn_zone = self.config.terrain.spawn_zone
        scene_type = self.config.terrain.scene_type

        if scene_type == 'lane':
            ny_zones = 1
            spawn_zone = (0, 0)

        grid = np.full((nx, ny), self.base_h)
        stair_mask = np.zeros((nx, ny), dtype=bool)
        
        # Apply global roughness unless the primary terrain type is perfectly flat
        if self.config.terrain.terrain_type != 'flat':
            self._apply_roughness(grid, np_random)

        zx_size, zy_size = nx // nx_zones, ny // ny_zones
        default_faces = 4 if scene_type == 'arena' else 2

        for rx in range(nx_zones):
            for cy in range(ny_zones):
                r_s, r_e = rx * zx_size, (rx + 1) * zx_size
                c_s, c_e = cy * zy_size, (cy + 1) * zy_size
                zone = grid[r_s:r_e, c_s:c_e]

                # Clear the spawn zone
                if rx == spawn_zone[0] and cy == spawn_zone[1]:
                    self._clear_area(zone, m_per_px, faces=default_faces)
                    continue

                choice = self._get_terrain_type(np_random)
                is_stairs = self._fill_zone_with_feature(
                    zone, choice, np_random, step_depth_px, m_per_px, default_faces
                )
                if is_stairs:
                    stair_mask[r_s:r_e, c_s:c_e] = True
        
        # Smooth the seams between all the randomized zones
        grid = self._smooth_seams(grid, nx_zones, ny_zones)
        return grid, stair_mask

    def update_hfield(self, model: mujoco.MjModel, hfield_id: int, np_random: np.random.Generator) -> None:
        """
        Generates and injects terrain into MuJoCo

        Args:
            model: The MuJoCo model to update.
            hfield_id: Pre-cached ID of the 'terrain' hfield.
            np_random: NumPy random generator
        """
        if not self.config.terrain.enabled or hfield_id == -1:
            return

        ny, nx = model.hfield_nrow[hfield_id], model.hfield_ncol[hfield_id]
        radius_x = model.hfield_size[hfield_id][0]
        m_per_px = (2.0 * radius_x) / nx
        step_depth_px = max(1, int(self.config.terrain.step_width_m / m_per_px))

        scene_type = self.config.terrain.scene_type
        mode = 'grid' if scene_type == 'arena' else self.config.terrain.curriculum_mode

        if mode == 'grid':
            grid, stair_mask = self.generate_grid_terrain(np_random, nx, ny, step_depth_px, m_per_px)
        elif mode == 'single':
            grid = np.full((nx, ny), self.base_h)
            stair_mask = np.zeros((nx, ny), dtype=bool)
            choice = self._get_terrain_type(np_random)
            f = np_random.choice([1, 2])

            is_stairs = self._fill_zone_with_feature(
                grid, choice, np_random, step_depth_px, m_per_px, faces=f
            )
            if is_stairs:
                stair_mask[:] = True
        else:
            print(f"Terrain mode '{mode}' not recognized. Generating flat floor.")
            grid = np.full((nx, ny), self.base_h)
            stair_mask = np.zeros((nx, ny), dtype=bool)

        lock_size_m = self.config.terrain.spawn_lock_size_m
        lock_blend_m = self.config.terrain.spawn_lock_blend_m

        if scene_type == 'lane': 
            lock_dist = np.arange(nx)[:, None] * m_per_px
            lock_mask = np.clip(((lock_size_m + 1.0) - lock_dist) / lock_blend_m, 0, 1)
            # Apply S-curve to spawn lock
            lock_mask = 0.5 * (1.0 - np.cos(lock_mask * np.pi))
            grid[:] = self.base_h + (grid - self.base_h) * (1.0 - lock_mask)
        else:
            r = np.arange(nx); c = np.arange(ny)
            X, Y = np.meshgrid((r - nx//2) * m_per_px, (c - ny//2) * m_per_px, indexing='ij')
            d = np.sqrt(X**2 + Y**2)
            lock_mask = np.clip((lock_size_m - d) / lock_blend_m, 0, 1)
            # Apply S-curve to spawn lock
            lock_mask = 0.5 * (1.0 - np.cos(lock_mask * np.pi))
            grid[:] = self.base_h + (grid - self.base_h) * (1.0 - lock_mask)

        # Global Smoothing Pass
        smoothed_grid = scipy.ndimage.gaussian_filter(grid, sigma=self.config.terrain.global_sigma)

        # Soft-Masking for Stairs:
        # We want stairs to stay relatively sharp, but their EDGES should blend
        # into the smoothed ground. We blur the stair_mask to create a 'fade' effect.
        soft_stair_mask = scipy.ndimage.gaussian_filter(stair_mask.astype(float), sigma=2.0)

        # result = (sharp_stairs * mask) + (smooth_ground * (1-mask))
        final_grid = (soft_stair_mask * grid) + (1.0 - soft_stair_mask) * smoothed_grid

        # Normalize to [0, 1] range based on elevation_z
        elevation_z = model.hfield_size[hfield_id][2]
        model.hfield_data[:] = (final_grid.T / elevation_z).flatten()

    def save(self, grid: np.ndarray, filename: str) -> None:
        """
        Saves the terrain to a file. Supports .npz for data and images for visualization

        Args:
            grid: The 2D terrain array to save
            filename: Output path. Extension determines format (.npz, .png, .jpg)
        """
        ext = filename.split('.')[-1].lower()

        if ext == 'npz':
            np.savez_compressed(filename, terrain=grid)
            print(f'Terrain data saved to {filename}')
        else:
            plt.figure(figsize=(12, 6))
            plt.imshow(grid.T, origin='lower', cmap='terrain')
            plt.colorbar(label='Height (meters)')
            plt.title(f'Procedural Terrain: {filename}')
            plt.xlabel('X Index (Forward)')
            plt.ylabel('Y Index (Side)')
            plt.savefig(filename)
            plt.close()
            print(f'Terrain image saved to {filename}')

    def load(self, filename: str) -> np.ndarray:
        """
        Loads terrain from a file. Supports .npz and image formats

        Args:
            filename: Path to the terrain file

        Returns:
            A 2D numpy array of heights
        """
        ext = filename.split('.')[-1].lower()
        if ext == 'npz':
            data = np.load(filename)
            return data['terrain']
        else:
            img = plt.imread(filename)
            if len(img.shape) == 3:
                img = np.mean(img[:, :, :3], axis=-1)
            return img.T

    def get_height_at(self, model: mujoco.MjModel, hfield_id: int, x: float, y: float) -> float:
        """
        Samples the terrain height at a specific world coordinate (x, y)

        Args:
            model: MuJoCo model containing hfield data
            hfield_id: Pre-cached ID of the 'terrain' hfield.
            x, y: World coordinates

        Returns:
            The interpolated height in meters
        """
        if hfield_id == -1:
            return 0.0

        radius_x, radius_y, elevation_z, _ = model.hfield_size[hfield_id]
        ny = model.hfield_nrow[hfield_id]
        nx = model.hfield_ncol[hfield_id]

        # Get HField geom world position
        geom_pos = [0, 0, 0]
        for i in range(model.ngeom):
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_HFIELD:
                geom_pos = model.geom_pos[i]
                break

        # Map world (x, y) to local hfield indices [0, 1]
        local_x = (x - geom_pos[0] + radius_x) / (2.0 * radius_x)
        local_y = (y - geom_pos[1] + radius_y) / (2.0 * radius_y)

        # Clip to boundaries
        local_x, local_y = np.clip(local_x, 0, 1), np.clip(local_y, 0, 1)

        # Convert to pixel indices
        ix = int(local_x * (nx - 1))
        iy = int(local_y * (ny - 1))

        # Retrieve data (MuJoCo buffer is row-major Y, then X)
        # In newer MuJoCo versions, this is already normalized [0, 1] relative to elevation_z
        height_raw = model.hfield_data[iy * nx + ix]

        # Scale back to meters
        # height = (hfield_data * elevation_z) + pos_z
        return height_raw * elevation_z + geom_pos[2]
