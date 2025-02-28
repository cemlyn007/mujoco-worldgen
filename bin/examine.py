#!/usr/bin/env python3
import logging

import click

from mujoco_worldgen.util.envs import EnvViewer, examine_env
from mujoco_worldgen.util.parse_arguments import parse_arguments
from mujoco_worldgen.util.path import worldgen_path

logger = logging.getLogger(__name__)


# For more detailed on argv information, please have a look on
# docstring below.
@click.command()
@click.argument('argv', nargs=-1, required=False)
def main(argv):
    '''
    examine.py is used to display environments

    Example uses:
        bin/examine.py simple_particle
        bin/examine.py examples/particle_gather.py
        bin/examine.py particle_gather n_food=5 floorsize=5
        bin/examine.py example_env_examine.jsonnet
    '''
    env_names, env_kwargs = parse_arguments(argv)
    assert len(env_names) == 1, 'You must provide exactly 1 environment to examine.'
    env_name = env_names[0]
    examine_env(env_name, env_kwargs,
                core_dir=worldgen_path(), envs_dir='examples', xmls_dir='xmls',
                env_viewer=EnvViewer)

    print(main.__doc__)


if __name__ == '__main__':
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    main()
