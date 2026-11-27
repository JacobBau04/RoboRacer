"""Static-map clearance queries, independent of ROS and the learned-model code."""
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import yaml


class WallEnv:
    def __init__(self, map_yaml):
        path = Path(map_yaml).expanduser().resolve()
        config = yaml.safe_load(path.read_text())
        # Image rows run opposite to map y.
        image = np.flipud(np.asarray(Image.open(path.parent / config['image']).convert('L')))
        probability = image / 255.0 if config.get('negate', 0) else (255.0 - image) / 255.0
        # Unknown pixels are not certified free. Pad the map with occupied cells.
        free = probability < config.get('free_thresh', 0.196)
        self.wall_resolution = float(config['resolution'])
        if abs(float(config['origin'][2])) > 1e-9:
            raise ValueError('Rotated occupancy maps are not supported')
        sdf = distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * self.wall_resolution
        self.wall_sdf = jnp.asarray(sdf, dtype=jnp.float32)
        self.wall_origin = jnp.asarray(config['origin'][:2], dtype=jnp.float32)

    def sample_wall_distance(self, xy):
        """Conservative cell clearance; outside/unknown is occupied, never clipped to free."""
        cell = jnp.floor((xy - self.wall_origin) / self.wall_resolution).astype(jnp.int32)
        x, y = cell[..., 0], cell[..., 1]
        height, width = self.wall_sdf.shape
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        distance = self.wall_sdf[jnp.clip(y, 0, height - 1), jnp.clip(x, 0, width - 1)]
        # Account for distance-field discretization and occupied-cell area.
        return jnp.where(inside, jnp.maximum(0.0, distance - 1.415 * self.wall_resolution), 0.0)
