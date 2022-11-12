import threading
from xml.dom import minidom
import copy
import hashlib
import inspect
import io
import logging
import os
from numbers import Number
from typing import Optional, Callable, Union
import tempfile
import gym
import mujoco
import numpy as np
from gym.spaces import Box, Tuple, Dict

from mujoco_worldgen.util.sim_funcs import (
    empty_get_info,
    flatten_get_obs,
    false_get_diverged,
    ctrl_set_action,
    zero_get_reward,
)
from mujoco_worldgen.util.types import enforce_is_callable
import collections
logger = logging.getLogger(__name__)


_MjSim_render_lock = threading.Lock()

class SimState(collections.namedtuple('SimStateBase', 'time qpos qvel act udd_state')):
    """Represents a snapshot of the simulator's state.
    This includes time, qpos, qvel, act, and udd_state.
    https://github.com/openai/mujoco-py/blob/master/mujoco_py/Simstate.pyx
    """
    __slots__ = ()

    # need to implement this because numpy doesn't support == on arrays
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        if set(self.udd_state.keys()) != set(other.udd_state.keys()):
            return False

        for k in self.udd_state.keys():
            if isinstance(self.udd_state[k], Number) and self.udd_state[k] != other.udd_state[k]:
                return False
            elif not np.array_equal(self.udd_state[k], other.udd_state[k]):
                return False

        return (self.time == other.time and
                np.array_equal(self.qpos, other.qpos) and
                np.array_equal(self.qvel, other.qvel) and
                np.array_equal(self.act, other.act))

    def __ne__(self, other):
        return not self.__eq__(other)

    def flatten(self):
        """ Flattens a state into a numpy array of numbers."""
        if self.act is None:
            act = np.empty(0)
        else:
            act = self.act
        state_tuple = ([self.time], self.qpos, self.qvel, act,
                       SimState._flatten_dict(self.udd_state))
        return np.concatenate(state_tuple)

    @staticmethod
    def _flatten_dict(d):
        a = []
        for k in sorted(d.keys()):
            v = d[k]
            if isinstance(v, Number):
                a.extend([v])
            else:
                a.extend(v.ravel())

        return np.array(a)

    @staticmethod
    def from_flattened(array, sim):
        idx_time = 0
        idx_qpos = idx_time + 1
        idx_qvel = idx_qpos + sim.model.nq
        idx_act = idx_qvel + sim.model.nv
        idx_udd = idx_act + sim.model.na

        time = array[idx_time]
        qpos = array[idx_qpos:idx_qpos + sim.model.nq]
        qvel = array[idx_qvel:idx_qvel + sim.model.nv]
        if sim.model.na == 0:
            act = None
        else:
            act = array[idx_act:idx_act + sim.model.na]
        flat_udd_state = array[idx_udd:]
        udd_state = SimState._unflatten_dict(flat_udd_state, sim.udd_state)

        return SimState(time, qpos, qvel, act, udd_state)

    @staticmethod
    def _unflatten_dict(a, schema_example):
        d = {}
        idx = 0
        for k in sorted(schema_example.keys()):
            schema_val = schema_example[k]
            if isinstance(schema_val, Number):
                val = a[idx]
                idx += 1
                d[k] = val
            else:
                assert isinstance(schema_val, np.ndarray)
                val_array = a[idx:idx + schema_val.size]
                idx += schema_val.size
                val = np.array(val_array).reshape(schema_val.shape)
                d[k] = val
        return d


class Sim:
    """https://github.com/openai/mujoco-py/blob/master/mujoco_py/Sim.pyx"""

    def __init__(self, model: mujoco.MjModel, data: Optional[mujoco.MjData] = None, nsubsteps: int = 1,
                 udd_callback: Callable=None, substep_callback: Callable=None, render_callback: Callable=None) -> None:
        self.model = model
        self.data = mujoco.MjData(self.model) if data is None else data
        self.nsubsteps = nsubsteps
        self.udd_state = None
        self._udd_callback = udd_callback
        if substep_callback is None:
            self.substep_callback = lambda: None
        else:
            self.substep_callback = substep_callback

        self.render_contexts = []
        self._render_context_offscreen = None
        self._render_context_window = None
        self.render_callback = render_callback
        self.extras = {}

        self._body_names, self._body_name2id, self._body_id2name = self._extract_mj_names(
            self.model.name_bodyadr, self.model.nbody, mujoco.mjtObj.mjOBJ_BODY
        )
        self._joint_names, self._joint_name2id, self._joint_id2name = self._extract_mj_names(
            self.model.name_jntadr, self.model.njnt, mujoco.mjtObj.mjOBJ_JOINT
        )
        self._geom_names, self._geom_name2id, self._geom_id2name = self._extract_mj_names(
            self.model.name_geomadr, self.model.ngeom, mujoco.mjtObj.mjOBJ_GEOM
        )
        self._site_names, self._site_name2id, self._site_id2name = self._extract_mj_names(
            self.model.name_siteadr, self.model.nsite, mujoco.mjtObj.mjOBJ_SITE
        )

    @property
    def body_names(self) -> tuple[str]:
        return self._body_names

    @property
    def body_name2id(self) -> dict[str, int]:
        return self._body_name2id

    @property
    def body_id2name(self) -> dict[int, str]:
        return self._body_id2name

    @property
    def joint_names(self) -> tuple[str]:
        return self._joint_names

    @property
    def joint_name2id(self) -> dict[str, int]:
        return self._joint_name2id

    @property
    def joint_id2name(self) -> dict[int, str]:
        return self._joint_id2name

    @property
    def geom_names(self) -> tuple[str]:
        return self._geom_names

    @property
    def geom_name2id(self) -> dict[str, int]:
        return self._geom_name2id

    @property
    def geom_id2name(self) -> dict[int, str]:
        return self._geom_id2name

    @property
    def site_names(self) -> tuple[str]:
        return self._site_names

    @property
    def site_name2id(self) -> dict[str, int]:
        return self._site_name2id

    @property
    def site_id2name(self) -> dict[int, str]:
        return self._site_id2name

    def reset(self) -> None:
        """
        Resets the simulation data and clears buffers.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.udd_state = None
        self.step_udd()

    def forward(self) -> None:
        """
        Computes the forward kinematics. Calls ``mj_forward`` internally.
        """
        mujoco.mj_forward(self.model, self.data)

    def set_constants(self):
        """
        Set constant fields of mjModel, corresponding to qpos0 configuration.
        """
        mujoco.mj_setConst(self.model, self.data)

    def step(self, with_udd: bool=True) -> None:
        if with_udd:
            self.step_udd()
        for _ in range(self.nsubsteps):
            self.substep_callback()
            mujoco.mj_step(self.model, self.data)

    def render(self, width=None, height=None, *, camera_name=None, depth=False, mode='offscreen',
               device_id=-1, segmentation=False) -> None:
        """
        Renders view from a camera and returns image as an `numpy.ndarray`.
        Args:
        - width (int): desired image width.
        - height (int): desired image height.
        - camera_name (str): name of camera in model. If None, the free
            camera will be used.
        - depth (bool): if True, also return depth buffer
        - device (int): device to use for rendering (only for GPU-backed
            rendering).
        Returns:
        - rgb (uint8 array): image buffer from camera
        - depth (float array): depth buffer from camera (only returned
            if depth=True)
        """
        if camera_name is None:
            camera_id = None
        else:
            camera_id = self.model.camera_name2id(camera_name)

        if mode == 'offscreen':
            with _MjSim_render_lock:
                if self._render_context_offscreen is None:
                    from mujoco_worldgen.util.envs import mjviewer
                    render_context = mjviewer.MjRenderContextOffscreen(self, device_id=device_id)
                else:
                    render_context = self._render_context_offscreen

                render_context.render(width=width, height=height,
                                      camera_id=camera_id,
                                      segmentation=segmentation)
                return render_context.read_pixels(width, height,
                                                  depth=depth,
                                                  segmentation=segmentation)
        elif mode == 'window':
            if self._render_context_window is None:
                from mujoco_worldgen.util.envs import mjviewer
                render_context = mjviewer.MjViewer(self)
            else:
                render_context = self._render_context_window

            render_context.render()
        else:
            raise ValueError("Mode must be either 'window' or 'offscreen'.")

    def add_render_context(self, render_context):
        self.render_contexts.append(render_context)
        if render_context.offscreen and self._render_context_offscreen is None:
            self._render_context_offscreen = render_context
        elif not render_context.offscreen and self._render_context_window is None:
            self._render_context_window = render_context

    @property
    def udd_callback(self):
        return self._udd_callback

    def step_udd(self):
        if self._udd_callback is None:
            self.udd_state = {}
        else:
            schema_example = self.udd_state
            self.udd_state = self._udd_callback(self)
            # Check to make sure the udd_state has consistent keys and dimension across steps
            if schema_example is not None:
                keys = set(schema_example.keys()) | set(self.udd_state.keys())
                for key in keys:
                    assert key in schema_example, "Keys cannot be added to udd_state between steps."
                    assert key in self.udd_state, "Keys cannot be dropped from udd_state between steps."
                    if isinstance(schema_example[key], Number):
                        assert isinstance(self.udd_state[key], Number), \
                            "Every value in udd_state must be either a number or a numpy array"
                    else:
                        assert isinstance(self.udd_state[key], np.ndarray), \
                            "Every value in udd_state must be either a number or a numpy array"
                        assert self.udd_state[key].shape == schema_example[key].shape, \
                            "Numpy array values in udd_state must keep the same dimension across steps."

    @udd_callback.setter
    def udd_callback(self, value):
        self._udd_callback = value
        self.udd_state = None
        self.step_udd()

    def get_state(self) -> SimState:
        """ Returns a copy of the simulator state. """
        qpos = np.copy(self.data.qpos)
        qvel = np.copy(self.data.qvel)
        if self.model.na == 0:
            act = None
        else:
            act = np.copy(self.data.act)
        udd_state = copy.deepcopy(self.udd_state)

        return SimState(self.data.time, qpos, qvel, act, udd_state)

    def set_state(self, value: SimState) -> None:
        """
        Sets the state from an SimState.
        If the SimState was previously unflattened from a numpy array, consider
        set_state_from_flattened, as the defensive copy is a substantial overhead
        in an inner loop.
        Args:
        - value (SimState): the desired state.
        """
        self.data.time = value.time
        self.data.qpos[:] = np.copy(value.qpos)
        self.data.qvel[:] = np.copy(value.qvel)
        if self.model.na != 0:
            self.data.act[:] = np.copy(value.act)
        self.udd_state = copy.deepcopy(value.udd_state)

    def set_state_from_flattened(self, value):
        """ This helper method sets the state from an array without requiring a defensive copy."""
        state = SimState.from_flattened(value, self)

        self.data.time = state.time
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel
        if self.model.na != 0:
            self.data.act[:] = state.act
        self.udd_state = state.udd_state

    def save(self, file: io.IOBase, format='xml', keep_inertials=False):
        """
        Saves the simulator model and state to a file as either
        a MuJoCo XML or MJB file. The current state is saved as
        a keyframe in the model file. This is useful for debugging
        using MuJoCo's `simulate` utility.
        Note that this doesn't save the UDD-state which is
        part of SimState, since that's not supported natively
        by MuJoCo. If you want to save the model together with
        the UDD-state, you should use the `get_xml` or `get_mjb`
        methods on `MjModel` together with `Sim.get_state` and
        save them with e.g. pickle.
        Args:
        - file (IO stream): stream to write model to.
        - format: format to use (either 'xml' or 'mjb')
        - keep_inertials (bool): if False, removes all <inertial>
          properties derived automatically for geoms by MuJoco. Note
          that this removes ones that were provided by the user
          as well.
        """
        xml_str = self.model.get_xml()
        dom = minidom.parseString(xml_str)

        mujoco_node = dom.childNodes[0]
        assert mujoco_node.tagName == 'mujoco'

        keyframe_el = dom.createElement('keyframe')
        key_el = dom.createElement('key')
        keyframe_el.appendChild(key_el)
        mujoco_node.appendChild(keyframe_el)

        def str_array(arr):
            return " ".join(map(str, arr))

        key_el.setAttribute('time', str(self.data.time))
        key_el.setAttribute('qpos', str_array(self.data.qpos))
        key_el.setAttribute('qvel', str_array(self.data.qvel))
        if self.data.act is not None:
            key_el.setAttribute('act', str_array(self.data.act))

        if not keep_inertials:
            for element in dom.getElementsByTagName('inertial'):
                element.parentNode.removeChild(element)

        result_xml = "\n".join(stripped for line in dom.toprettyxml(indent=" " * 4).splitlines()
                               if (stripped := line.strip()))

        if format == 'xml':
            file.write(result_xml)
        elif format == 'mjb':
            new_model = mujoco.MjModel.from_xml_string(result_xml)
            file.write(new_model.get_mjb())
        else:
            raise ValueError("Unsupported format. Valid ones are 'xml' and 'mjb'")

    def ray(self, pnt, vec, include_static_geoms=True, exclude_body=-1, group_filter=None):
        """
        Cast a ray into the scene, and return the first valid geom it intersects.
            pnt - origin point of the ray in world coordinates (X Y Z)
            vec - direction of the ray in world coordinates (X Y Z)
            include_static_geoms - if False, we exclude geoms that are children of worldbody.
            exclude_body - if this is a body ID, we exclude all children geoms of this body.
            group_filter - a vector of booleans of length const.NGROUP
                           which specifies what geom groups (stored in model.geom_group)
                           to enable or disable.  If none, all groups are used
        Returns (distance, geomid) where
            distance - distance along ray until first collision with geom
            geomid - id of the geom the ray collided with
        If no collision was found in the scene, return (-1, None)
        NOTE: sometimes self.forward() needs to be called before self.ray().
        See self.ray_fast_group() and self.ray_fast_nogroup() for versions of this call
        with more stringent type requirements.
        """
        if group_filter is None:
            return self.ray_fast_nogroup(
                np.asarray(pnt, dtype=np.float64),
                np.asarray(vec, dtype=np.float64),
                1 if include_static_geoms else 0,
                exclude_body)
        else:
            return self.ray_fast_group(
                np.asarray(pnt, dtype=np.float64),
                np.asarray(vec, dtype=np.float64),
                np.asarray(group_filter, dtype=np.uint8),
                1 if include_static_geoms else 0,
                exclude_body)

    def ray_fast_group(self,
                       pnt: np.ndarray,
                       vec: np.ndarray,
                       geomgroup: np.ndarray,
                       flg_static: int = 1,
                       bodyexclude: int = -1) -> tuple[float, Optional[int]]:
        """
        Faster version of sim.ray(), which avoids extra copies,
        but needs to be given all the correct type arrays.
        See self.ray() for explanation of arguments
        """
        geomid = np.empty((1, 1), dtype=np.int32)
        distance = mujoco.mj_ray(self.model.ptr, self.data.ptr, pnt, vec, geomgroup, flg_static,
                                 bodyexclude, geomid)

        assert distance == -1 and geomid is not None
        collision_geom = geomid if geomid != -1 else None
        return (distance, collision_geom)

    def ray_fast_nogroup(self,
                         pnt: np.ndarray,
                         vec: np.ndarray,
                         flg_static: int = 1,
                         bodyexclude: int = -1) -> tuple[float, Optional[int]]:
        """
        Faster version of sim.ray(), which avoids extra copies,
        but needs to be given all the correct type arrays.
        This version hardcodes the geomgroup to NULL.
        See self.ray() for explanation of arguments
        """
        geomid = np.empty((1, 1), dtype=np.int32)
        distance = mujoco.mj_ray(self.model, self.data, pnt, vec, None, flg_static, bodyexclude,
                                 geomid)
        # TODO: Probably wrong.
        collision_geom = geomid if geomid != -1 else None
        assert distance == -1
        return (distance, collision_geom)

    def get_joint_qpos_addr(self, name: str) -> Union[int, tuple]:
        '''
        Returns the qpos address for given joint.
        Returns:
        - address (int, tuple): returns int address if 1-dim joint, otherwise
            returns the a (start, end) tuple for pos[start:end] access.
        '''
        joint_id = self.joint_name2id[name]
        joint_type = self.model.jnt_type[joint_id]
        joint_addr = self.model.jnt_qposadr[joint_id]

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 7
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 4
        else:
            assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE,
                                  mujoco.mjtJoint.mjJNT_SLIDE)
            ndim = 1

        if ndim == 1:
            return joint_addr
        else:
            return (joint_addr, joint_addr + ndim)

    def get_joint_qvel_addr(self, name: str) -> Union[int, tuple]:
        '''
        Returns the qvel address for given joint.
        Returns:
        - address (int, tuple): returns int address if 1-dim joint, otherwise
            returns the a (start, end) tuple for vel[start:end] access.
        '''
        joint_id = self.joint_name2id[name]
        joint_type = self.model.jnt_type[joint_id]
        joint_addr = self.model.jnt_qposadr[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 6
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 3
        else:
            assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            ndim = 1

        if ndim == 1:
            return joint_addr
        else:
            return (joint_addr, joint_addr + ndim)

    def _extract_mj_names(self, name_adr: list[int],
                          n: int, obj_type: mujoco.mjtObj) -> tuple[tuple, dict, dict]:
        # objects don't need to be named in the XML, so name might be None
        if len(name_adr) != n:
            raise ValueError
        # else...
        id2name = {i: None for i in range(n)}
        name2id = {}
        for i in range(n):
            name = self.model.names[name_adr[i]:].split(b'\x00')[0]
            decoded_name = name.decode()
            if decoded_name:
                obj_id = mujoco.mj_name2id(self.model, obj_type, decoded_name)
                assert 0 <= obj_id < n and id2name[obj_id] is None
                name2id[decoded_name] = obj_id
                id2name[obj_id] = decoded_name

        # sort names by increasing id to keep order deterministic
        return tuple(id2name[id_] for id_ in sorted(name2id.values())), name2id, id2name


class Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self,
                 get_sim,
                 get_obs=flatten_get_obs,
                 get_reward=zero_get_reward,
                 get_info=empty_get_info,
                 get_diverged=false_get_diverged,
                 set_action=ctrl_set_action,
                 action_space=None,
                 horizon=100,
                 start_seed=None,
                 deterministic_mode=False):
        """
        Env is a Gym environment subclass tuned for robotics learning
        research.

        Args:
        - get_sim (callable): a callable that returns a Sim.
        - get_obs (callable): callable with a Sim object as the sole
            argument and should return observations.
        - set_action (callable): callable which takes a Sim object and
            updates its data and buffer directly.
        - get_reward (callable): callable which takes a Sim object and
            returns a scalar reward.
        - get_info (callable): callable which takes a Sim object and
            returns info (dictionary).
        - get_diverged (callable): callable which takes a Sim object
            and returns a (bool, float) tuple. First value is True if
            simulator diverged and second value is the reward at divergence.
        - action_space: a space of allowed actions or a two-tuple of a ranges
            if number of actions is unknown until the simulation is instantiated
        - horizon (int): horizon of environment (i.e. max number of steps).
        - start_seed (int or string): seed for random state generator (None for random seed).
            Strings will be hashed.  A non-None value implies deterministic_mode=True.
            This argument allows us to run a deterministic series of goals/randomizations
            for a given policy.  Then applying the same seed to another policy will allow the
            comparison of results more accurately.  The reason a string is allowed is so
            that we can more easily find and share seeds that are farther from 0,
            which is the default starting point for deterministic_mode, and thus have
            more likelihood of getting a performant sequence of goals.
        """
        if (horizon is not None) and not isinstance(horizon, int):
            raise TypeError('horizon must be an int')

        self.get_sim = enforce_is_callable(get_sim, (
            'get_sim should be callable and should return a Sim object'))
        self.get_obs = enforce_is_callable(get_obs, (
            'get_obs should be callable with a Sim object as the sole '
            'argument and should return observations'))
        self.set_action = enforce_is_callable(set_action, (
            'set_action should be a callable which takes a Sim object and '
            'updates its data and buffer directly'))
        self.get_reward = enforce_is_callable(get_reward, (
            'get_reward should be a callable which takes a Sim object and '
            'returns a scalar reward'))
        self.get_info = enforce_is_callable(get_info, (
            'get_info should be a callable which takes a Sim object and '
            'returns a dictionary'))
        self.get_diverged = enforce_is_callable(get_diverged, (
            'get_diverged should be a callable which takes a Sim object '
            'and returns a (bool, float) tuple. First value is whether '
            'simulator is diverged (or done) and second value is the reward at '
            'that time.'))

        self.sim: Optional[Sim] = None
        self.horizon = horizon
        self.t = None
        self.deterministic_mode = deterministic_mode

        # Numpy Random State
        if isinstance(start_seed, str):
            start_seed = int(hashlib.sha1(start_seed.encode()).hexdigest(), 16) % (2**32)
            self.deterministic_mode = True
        elif isinstance(start_seed, int):
            self.deterministic_mode = True
        else:
            start_seed = 0 if self.deterministic_mode else np.random.randint(2**32)
        self._random_state = np.random.RandomState(start_seed)
        # Seed that will be used on next _reset()
        self._next_seed = start_seed
        # Seed that was used in last _reset()
        self._current_seed = None

        # For rendering
        self.viewer = None

        # These are required by Gym
        self._action_space = action_space
        self._observation_space = None
        self._spec = Spec(max_episode_steps=horizon, timestep_limit=horizon)
        self._name = None

    # This is to mitigate issues with old/new envs
    @property
    def unwrapped(self):
        return self

    @property
    def name(self):
        if self._name is None:
            name = str(inspect.getfile(self.get_sim))
            if name.endswith(".py"):
                name = name[:-3]
            self._name = name
        return self._name

    def set_state(self, state, call_forward=True):
        """
        Sets the state of the enviroment to the given value. It does not
        set time.

        Warning: This only sets the MuJoCo state by setting qpos/qvel
            (and the user-defined state "udd_state"). It doesn't set
            the state of objects which don't have joints.

        Args:
        - state (SimState): desired state.
        - call_forward (bool): if True, forward simulation after setting
            state.
        """
        if self.sim is None:
            raise EmptyEnvException(
                "You must call reset() or reset_to_state() before setting the "
                "state the first time")

        # Call forward to write out values in the MuJoCo data.
        # Note: if udd_callback is set on the Sim instance, then the
        # user will need to call forward() manually before calling step.
        self.sim.set_state(state)
        if call_forward:
            self.sim.forward()

    def get_state(self):
        """
        Returns a copy of the current environment state.

        Returns:
        - state (SimState): state of the environment's Sim object.
        """
        if self.sim is None:
            raise EmptyEnvException(
                "You must call reset() or reset_to_state() before accessing "
                "the state the first time")
        return self.sim.get_state()

    def get_xml(self):
        '''
        :return: full state of the simulator serialized as XML (won't contain
                 meshes, textures, and data information).
        '''
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'model.xml')
            mujoco.mj_saveLastXML(filepath, self.sim.model)
            with open(filepath) as f:
                xml = f.read()
        return xml

    def get_mjb(self):
        '''
        :return: full state of the simulator serialized as mjb.
        '''
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'model.mjb'), 'wr') as f:
                mujoco.mj_saveModel(self.sim.model, f)
                f.seek(0, io.SEEK_SET)
                mjb = f.read()
        return mjb

    def reset_to_state(self, state, call_forward=True):
        """
        Reset to given state.

        Args:
        - state (SimState): desired state.
        """
        if not isinstance(state, SimState):
            raise TypeError(
                "You must reset to an explicit state (SimState).")

        if self.sim is None:
            if self._current_seed is None:
                self._update_seed()

            self.sim = self.get_sim(self._current_seed)
        else:
            # Ensure environment state not captured in MuJoCo's qpos/qvel
            # is reset to the state defined by the model.
            self.sim.reset()

        self.set_state(state, call_forward=call_forward)

        self.t = 0
        return self._reset_sim_and_spaces()

    def _update_seed(self, force_seed=None):
        if force_seed is not None:
            self._next_seed = force_seed
        self._current_seed = self._next_seed
        assert self._current_seed is not None
        # if in deterministic mode, then simply increment seed, otherwise randomize
        if self.deterministic_mode:
            self._next_seed = self._next_seed + 1
        else:
            self._next_seed = np.random.randint(2**32)
        # immediately update the seed in the random state object
        self._random_state.seed(self._current_seed)

    @property
    def current_seed(self):
        # Note: this is a property rather than just instance variable
        # for legacy and backwards compatibility reasons.
        return self._current_seed

    def _reset_sim_and_spaces(self):
        obs = self.get_obs(self.sim)

        # Mocaps are defined by 3-dim position and 4-dim quaternion
        if isinstance(self._action_space, tuple):
            assert len(self._action_space) == 2
            self._action_space = Box(
                self._action_space[0], self._action_space[1],
                (self.sim.model.nmocap * 7 + self.sim.model.nu, ), np.float32)
        elif self._action_space is None:
            self._action_space = Box(
                -np.inf, np.inf, (self.sim.model.nmocap * 7 + self.sim.model.nu, ), np.float32)
        self._action_space.flatten_dim = np.prod(self._action_space.shape)

        self._observation_space = gym_space_from_arrays(obs)
        if self.viewer is not None:
            self.viewer.update_sim(self.sim)

        return obs

    #
    # Custom pickling
    #

    def __getstate__(self):
        excluded_attrs = frozenset(
            ("sim", "viewer", "_monitor"))
        attr_values = {k: v for k, v in self.__dict__.items()
                       if k not in excluded_attrs}
        if self.sim is not None:
            attr_values['sim_state'] = self.get_state()
        return attr_values

    def __setstate__(self, attr_values):
        for k, v in attr_values.items():
            if k != 'sim_state':
                self.__dict__[k] = v

        self.sim = None
        self.viewer = None
        if 'sim_state' in attr_values:
            if self.sim is None:
                assert self._current_seed is not None
                self.sim = self.get_sim(self._current_seed)
            self.set_state(attr_values['sim_state'])
            self._reset_sim_and_spaces()

        return self

    def logs(self):
        logs = []
        if hasattr(self.env, 'logs'):
            logs += self.env.logs()
        return logs

    #
    # GYM REQUIREMENTS: these are methods required to be compatible with Gym
    #

    @property
    def action_space(self):
        if self._action_space is None:
            raise EmptyEnvException(
                "You have to reset environment before accessing action_space.")
        return self._action_space

    @property
    def observation_space(self):
        if self._observation_space is None:
            raise EmptyEnvException(
                "You have to reset environment before accessing "
                "observation_space.")
        return self._observation_space

    def reset(self, force_seed: Optional[int]=None) -> tuple[dict, dict]:
        self._update_seed(force_seed=force_seed)

        # get sim with current seed
        self.sim = self.get_sim(self._current_seed)

        # init sim
        self.sim.forward()
        self.t = 0
        self.sim.data.time = 0.0
        return self._reset_sim_and_spaces(), {}

    def seed(self, seed: Optional[int]=None):
        """
        Use `env.seed(some_seed)` to set the seed that'll be used in
        `env.reset()`. More specifically, this is the seed that will
        be passed into `env.get_sim` during `env.reset()`. The seed
        will then be incremented in consequent calls to `env.reset()`.
        For example:

            env.seed(0)
            env.reset() -> gives seed(0) world
            env.reset() -> gives seed(1) world
            ...
            env.seed(0)
            env.reset() -> gives seed(0) world
        """
        if isinstance(seed, list):
            # Support list of seeds as required by Gym.
            assert len(seed) == 1, "Only a single seed supported."
            self._next_seed = seed[0]
        elif isinstance(seed, int):
            self._next_seed = seed
        elif seed is not None:
            # If seed is None, we just return current seed.
            raise ValueError("Seed must be an integer.")

        # Return list of seeds to conform to Gym specs
        return [self._next_seed]

    def step(self, action):
        action = np.asarray(action)
        action = np.minimum(action, self.action_space.high)
        action = np.maximum(action, self.action_space.low)
        assert self.action_space.contains(action), (
            'Action should be in action_space:\nSPACE=%s\nACTION=%s' %
            (self.action_space, action)
        )
        self.set_action(self.sim, action)
        self.sim.step()
        # Need to call forward() so that sites etc are updated,
        # since they're used in the reward computations.
        self.sim.forward()
        self.t += 1

        reward = self.get_reward(self.sim)
        if not isinstance(reward, float):
            raise TypeError("The return value of get_reward must be a float")

        obs = self.get_obs(self.sim)
        diverged, divergence_reward = self.get_diverged(self.sim)

        if not isinstance(diverged, bool):
            raise TypeError(
                "The first return value of get_diverged must be boolean")
        if not isinstance(divergence_reward, float):
            raise TypeError(
                "The second return value of get_diverged must be float")

        terminated = False
        truncated = False
        if diverged:
            terminated = True
            if divergence_reward is not None:
                reward = divergence_reward
        elif self.horizon is not None:
            truncated = (self.t >= self.horizon)

        info = self.get_info(self.sim)
        info["diverged"] = divergence_reward
        # Return value as required by Gym
        return obs, reward, terminated, truncated, info

    def observe(self):
        """ Gets a new observation from the environment. """
        self.sim.forward()
        return self.get_obs(self.sim)

    def render(self, mode='human', close=False):
        raise ValueError("Unsupported mode %s" % mode)


class EmptyEnvException(Exception):
    pass

# Helpers
###############################################################################


class Spec(object):
    # required by gym.wrappers.Monitor

    def __init__(self, max_episode_steps=np.inf, timestep_limit=np.inf):
        self.id = "worldgen.env"
        self.max_episode_steps = max_episode_steps
        self.timestep_limit = timestep_limit


def gym_space_from_arrays(arrays):
    if isinstance(arrays, np.ndarray):
        ret = Box(-np.inf, np.inf, arrays.shape, np.float32)
        ret.flatten_dim = np.prod(ret.shape)
    elif isinstance(arrays, (tuple, list)):
        ret = Tuple([gym_space_from_arrays(arr) for arr in arrays])
    elif isinstance(arrays, dict):
        ret = Dict(dict([(k, gym_space_from_arrays(v)) for k, v in arrays.items()]))
    else:
        raise TypeError("Array is of unsupported type.")
    return ret
