from math import sqrt

import numpy as np
import mujoco


def raycast(sim, geom1_id=None, geom2_id=None, pt1=None, pt2=None, geom_group=None):
    '''
        Given a mujoco sim, from a geom to a geom, point to a point
        Args:
            sim: Mujoco sim object
            geom1 (int), id of geom ray originates from
            geom2 (int), id of geom ray points to
            p1 (np.ndarray[3]): 3D point ray originates from
            p2 (np.ndarray[3]): 3D point ray points to
            geom_group: one-hot list determining which of the five geom groups
                        should be visible to the raycast
    '''
    assert (geom1_id is None) != (pt1 is None), "geom1_id or p1 must be specified"
    assert (geom2_id is None) != (pt2 is None), "geom2_id or p2 must be specified"
    if geom1_id is not None:
        pt1 = sim.data.geom_xpos[geom1_id]
        body1 = sim.model.geom_bodyid[geom1_id]
    else:
        # Don't exclude any bodies if we originate ray from a point
        body1 = np.max(sim.model.geom_bodyid) + 1
    if geom2_id is not None:
        pt2 = sim.data.geom_xpos[geom2_id]

    ray_direction = pt2 - pt1
    ray_direction /= sqrt(ray_direction[0] ** 2 + ray_direction[1] ** 2 + ray_direction[2] ** 2)

    if geom_group is not None:
        geom_group = np.array(geom_group, dtype=np.uint8)
    else:
        geom_group = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)  # This is the default geom group

    # Setup int array
    geomid = np.empty((1, 1), dtype=np.int32)
    flg_static = np.zeros((1, 1), dtype=np.uint8)
    dist = mujoco.mj_ray(sim.model,
                         sim.data,
                         pt1,
                         ray_direction,
                         geom_group,
                         flg_static,  # flg_static. TODO idk what this is
                         body1,  # Bodyid to exclude
                         geomid)
    collision_geom = geomid.item() if geomid != -1 else None
    return dist, collision_geom
