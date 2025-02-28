

import numpy as np

from mujoco_worldgen.objs.obj import Obj
from mujoco_worldgen.util.types import store_args


class Floor(Obj):
    '''
    Floor() is essentially a special box geom used as the base of experiments.
    It has no joints, so is essentially an immovable object.
    Placement is calculated in a fixed position, and encoded in XML,
        as opposed to in qpos, which other objects use.
    '''
    @store_args
    def __init__(self, geom_type='plane'):
        super(Floor, self).__init__()

    def generate(self, random_state, world_params, placement_size):
        top = dict(origin=(0, 0, 0), size=placement_size)
        self.placements = dict(top=top)
        self.size = np.array([placement_size[0], placement_size[1], 0.0])

    def generate_xml_dict(self):
        # Last argument in size is visual mesh resolution (it's not height).
        # keep it high if you want rendering to be fast.
        pos = self.absolute_position
        pos[0] += self.size[0] / 2.0
        pos[1] += self.size[1] / 2.0
        geom = dict()

        geom['@name'] = self.name
        geom['@pos'] = pos
        if self.geom_type == 'box':
            geom['@size'] = np.array([self.size[0] / 2.0, self.size[1] / 2.0, 0.000001])
            geom['@type'] = 'box'
        elif self.geom_type == 'plane':
            geom['@size'] = np.array([self.size[0] / 2.0, self.size[1] / 2.0, 1.0])
            geom['@type'] = 'plane'
        else:
            raise ValueError("Invalid geom_type: " + self.geom_type)
        geom['@condim'] = 3
        geom['@name'] = self.name

        # body is necessary to place sites.
        body = dict()
        body["@name"] = self.name
        body["@pos"] = pos

        worldbody = dict([("geom", [geom]),
                                 ("body", [body])])

        xml_dict = dict(worldbody=worldbody)
        return xml_dict
