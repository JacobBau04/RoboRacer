import numpy as np
from PIL import Image
import jax.numpy as jnp
import pytest

from wall_map import WallEnv


def test_occupied_unknown_and_outside_map_are_not_free(tmp_path):
    image = np.full((30, 30), 255, np.uint8)
    image[14:17, 14:17] = 0
    image[5:8, 5:8] = 205  # unknown in common ROS maps
    Image.fromarray(image).save(tmp_path / 'map.pgm')
    (tmp_path / 'map.yaml').write_text('image: map.pgm\nresolution: 0.1\norigin: [0, 0, 0]\n')
    env = WallEnv(tmp_path / 'map.yaml')
    points = jnp.array([[1.55, 1.45], [.65, 2.35], [-.1, 1.], [3.1, 1.]])
    assert np.all(np.asarray(env.sample_wall_distance(points)) == 0)
    assert float(env.sample_wall_distance(jnp.array([.75, .75]))) > .3


def test_rotated_map_fails_explicitly(tmp_path):
    Image.fromarray(np.full((3, 3), 255, np.uint8)).save(tmp_path / 'map.pgm')
    (tmp_path / 'map.yaml').write_text('image: map.pgm\nresolution: 0.1\norigin: [0, 0, 0.2]\n')
    with pytest.raises(ValueError, match='Rotated'):
        WallEnv(tmp_path / 'map.yaml')
