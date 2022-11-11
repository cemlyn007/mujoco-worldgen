"""https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/mjviewer.py"""
import abc
import copy
import os
import sys
import time
from multiprocessing import Process, Queue
from threading import Lock

import glfw
import imageio
import mujoco
import numpy as np


class OpenGLContext(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def make_context_current(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_buffer_size(self, width, height):
        raise NotImplementedError()


class GlfwError(RuntimeError):
    pass


class GlfwContext(OpenGLContext):
    _INIT_WIDTH = 1000
    _INIT_HEIGHT = 1000
    _GLFW_IS_INITIALIZED = False

    def __init__(self, offscreen=False, quiet=False):
        GlfwContext._init_glfw()

        self._width = self._INIT_WIDTH
        self._height = self._INIT_HEIGHT
        self.window = self._create_window(offscreen, quiet=quiet)
        self._set_window_size(self._width, self._height)

    @staticmethod
    def _init_glfw():
        if GlfwContext._GLFW_IS_INITIALIZED:
            return

        if 'glfw' not in globals():
            raise GlfwError("GLFW not installed")

        glfw.set_error_callback(GlfwContext._glfw_error_callback)

        # HAX: sometimes first init() fails, while second works fine.
        glfw.init()
        if not glfw.init():
            raise GlfwError("Failed to initialize GLFW")

        GlfwContext._GLFW_IS_INITIALIZED = True

    def make_context_current(self):
        glfw.make_context_current(self.window)

    def set_buffer_size(self, width, height):
        self._set_window_size(width, height)
        self._width = width
        self._height = height

    def _create_window(self, offscreen, quiet=False):
        if offscreen:
            if not quiet:
                print("Creating offscreen glfw")
            glfw.window_hint(glfw.VISIBLE, 0)
            glfw.window_hint(glfw.DOUBLEBUFFER, 0)
            init_width, init_height = self._INIT_WIDTH, self._INIT_HEIGHT
        else:
            if not quiet:
                print("Creating window glfw")
            glfw.window_hint(glfw.SAMPLES, 4)
            glfw.window_hint(glfw.VISIBLE, 1)
            glfw.window_hint(glfw.DOUBLEBUFFER, 1)
            resolution, _, refresh_rate = glfw.get_video_mode(
                glfw.get_primary_monitor())
            init_width, init_height = resolution

        self._width = init_width
        self._height = init_height
        window = glfw.create_window(self._width, self._height, "mujoco_py", None, None)

        if not window:
            raise GlfwError("Failed to create GLFW window")

        return window

    def get_buffer_size(self):
        return glfw.get_framebuffer_size(self.window)

    def _set_window_size(self, target_width, target_height):
        self.make_context_current()
        if target_width != self._width or target_height != self._height:
            self._width = target_width
            self._height = target_height
            glfw.set_window_size(self.window, target_width, target_height)

            # HAX: When running on a Mac with retina screen, the size
            # sometimes doubles
            width, height = glfw.get_framebuffer_size(self.window)
            if target_width != width and "darwin" in sys.platform.lower():
                glfw.set_window_size(self.window, target_width // 2, target_height // 2)

    @staticmethod
    def _glfw_error_callback(error_code, description):
        print("GLFW error (code %d): %s", error_code, description)


def rec_assign(node, assign):
    # Assigns values to node recursively.
    # This is neccessary to avoid overriding pointers in MuJoCo.
    for field in dir(node):
        if field.find("__") == -1 and field != 'uintptr':
            val = getattr(node, field)
            if isinstance(val, (int, bool, float, None.__class__, str)):
                setattr(node, field, assign[field])
            elif isinstance(val, np.ndarray):
                val[:] = assign[field][:]
            elif not hasattr(val, "__call__"):
                rec_assign(val, assign[field])


def rec_copy(node):
    # Recursively copies object to dictionary.
    # Applying directly copy.deepcopy causes seg fault.
    ret = {}
    for field in dir(node):
        if field.find("__") == -1:
            val = getattr(node, field)
            if isinstance(val, (int, bool, float, None.__class__, str)):
                ret[field] = val
            elif isinstance(val, np.ndarray):
                ret[field] = copy.deepcopy(val)
            elif not hasattr(val, "__call__"):
                ret[field] = rec_copy(val)
    return ret


class MjRenderContext:
    """
    Class that encapsulates rendering functionality for a
    MuJoCo simulation.
    """

    def __init__(self, sim, offscreen: bool = True, device_id: int = -1, opengl_backend=None, quiet=False):
        maxgeom = 1000
        self.sim = sim  # TODO: Is the model already attached?
        self._model = self.sim.model
        self._scn = mujoco.MjvScene(self._model, maxgeom)
        self.cam = mujoco.MjvCamera()
        self._pert = mujoco.MjvPerturb()
        self._vopt = mujoco.MjvOption()
        self._con = mujoco.MjrContext()

        self._setup_opengl_context(offscreen, device_id, opengl_backend, quiet=quiet)
        self.offscreen = offscreen

        # Ensure the model data has been updated so that there
        # is something to render
        sim.forward()

        sim.add_render_context(self)

        self._pert.active = 0
        self._pert.select = 0
        self._pert.skinselect = -1

        self._markers = []
        self._overlay = {}

        self._init_camera(sim)
        self._set_mujoco_buffers()

    def update_sim(self, new_sim):
        if new_sim == self.sim:
            return
        self._model = new_sim.model
        self._data = new_sim.data
        self._set_mujoco_buffers()
        for render_context in self.sim.render_contexts:
            new_sim.add_render_context(render_context)
        self.sim = new_sim

    def _set_mujoco_buffers(self):
        self._con = mujoco.MjrContext(self._model, mujoco.mjtFontScale.mjFONTSCALE_150)
        if self.offscreen:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._con)
            if self._con.currentBuffer != mujoco.mjtFramebuffer.mjFB_OFFSCREEN:
                raise RuntimeError('Offscreen rendering not supported')
        else:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self._con)
            if self._con.currentBuffer != mujoco.mjtFramebuffer.mjFB_WINDOW:
                raise RuntimeError('Window rendering not supported')
            # TODO: Perhaps this is here for resource management later?
            # self.con = WrapMjrContext(self._con)

    def _setup_opengl_context(self, offscreen, device_id, opengl_backend, quiet=False):
        if opengl_backend is None and (not offscreen or sys.platform == 'darwin'):
            # default to glfw for onscreen viewing or mac (both offscreen/onscreen)
            opengl_backend = 'glfw'

        if opengl_backend == 'glfw':
            self.opengl_context = GlfwContext(offscreen=offscreen, quiet=quiet)
        else:
            if device_id < 0:
                if "GPUS" in os.environ:
                    device_id = os.environ["GPUS"]
                else:
                    device_id = os.getenv('CUDA_VISIBLE_DEVICES', '')
                if len(device_id) > 0:
                    device_id = int(device_id.split(',')[0])
                else:
                    # Sometimes env variable is an empty string.
                    device_id = 0
            # TODO: Might cause issues not having offscreen rendering.
            self.opengl_context = None  # OffscreenOpenGLContext(device_id)

    def _init_camera(self, sim):
        # Make the free camera look at the scene
        self.cam.type = mujoco.mjtGridPos.CAMERA_FREE
        self.cam.fixedcamid = -1
        for i in range(3):
            self.cam.lookat[i] = np.median(sim.data.geom_xpos[:, i])
        self.cam.distance = sim.model.stat.extent

    def update_offscreen_size(self, width, height):
        if width != self._con.offWidth or height != self._con.offHeight:
            self._model.vis.global_.offwidth = width
            self._model.vis.global_.offheight = height
            self._con.free()
            self._set_mujoco_buffers()

    def render(self, width, height, camera_id=None, segmentation=False):
        rect = mujoco.MjrRect(0, 0, width, height)

        if self.sim.render_callback is not None:
            self.sim.render_callback(self.sim, self)

        # Sometimes buffers are too small.
        if width > self._con.offWidth or height > self._con.offHeight:
            new_width = max(width, self._model.vis.global_.offwidth)
            new_height = max(height, self._model.vis.global_.offheight)
            self.update_offscreen_size(new_width, new_height)

        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id

        mujoco.mjv_updateScene(self._model, self._data, self._vopt, self._pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL,
                               self._scn)

        if segmentation:
            self._scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
            self._scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

        for marker_params in self._markers:
            self._add_marker_to_scene(marker_params)

        mujoco.mjr_render(rect, self._scn, self._con)
        for gridpos, (text1, text2) in self._overlay.items():
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, gridpos, rect,
                               text1.encode(), text2.encode(), self._con)

        if segmentation:
            self._scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
            self._scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0

    def read_pixels(self, width, height, depth=True, segmentation=False):
        rect = mujoco.MjrRect(0, 0, width, height)

        rgb_arr = np.zeros(3 * rect.width * rect.height, dtype=np.uint8)
        depth_arr = np.zeros(rect.width * rect.height, dtype=np.float32)

        mujoco.mjr_readPixels(rgb_arr.ravel(), depth_arr.ravel(), rect, self._con)
        rgb_img = rgb_arr.reshape(rect.height, rect.width, 3)

        ret_img = rgb_img
        if segmentation:
            seg_img = (rgb_img[:, :, 0] + rgb_img[:, :, 1] * (2 ** 8) + rgb_img[:, :, 2] * (2 ** 16))
            seg_img[seg_img >= (self._scn.ngeom + 1)] = 0
            seg_ids = np.full((self._scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)

            for i in range(self._scn.ngeom):
                geom = self._scn.geoms[i]
                if geom.segid != -1:
                    seg_ids[geom.segid + 1, 0] = geom.objtype
                    seg_ids[geom.segid + 1, 1] = geom.objid
            ret_img = seg_ids[seg_img]

        if depth:
            depth_img = depth_arr.reshape(rect.height, rect.width)
            return (ret_img, depth_img)
        else:
            return ret_img

    def read_pixels_depth(self, buffer):  # np.ndarray[np.float32_t, mode="c", ndim=2]
        """ Read depth pixels into a preallocated buffer """
        rect = mujoco.mjr_rectangle((0, 0, buffer.shape[1], buffer[0]), 0., 0., 0., 0.)
        mujoco.mjr_readPixels(None, buffer, rect, self._con)

    def upload_texture(self, tex_id: int):
        """ Uploads given texture to the GPU. """
        self.opengl_context.make_context_current()
        mujoco.mjr_uploadTexture(self._model, self._con, tex_id)

    def draw_pixels(self, image: np.ndarray, left: int, bottom: int):  # np.ndarray[np.uint8_t, ndim=3]
        """Draw an image into the OpenGL buffer."""
        viewport = mujoco.MjrRect(left, bottom, image.shape[1], image.shape[0])
        mujoco.mjr_drawPixels(image.ravel(), None, viewport, self._con)

    def move_camera(self, action: int, reldx: float, reldy: float):
        """ Moves the camera based on mouse movements. Action is one of mjMOUSE_*. """
        mujoco.mjv_moveCamera(self._model, action, reldx, reldy, self._scn, self.cam)

    def add_overlay(self, gridpos: int, text1: str, text2: str):
        """ Overlays text on the scene. """
        if gridpos not in self._overlay:
            self._overlay[gridpos] = ["", ""]
        self._overlay[gridpos][0] += text1 + "\n"
        self._overlay[gridpos][1] += text2 + "\n"

    def add_marker(self, **marker_params):
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker_params):
        """ Adds marker to scene, and returns the corresponding object. """
        if self._scn.ngeom >= self._scn.maxgeom:
            raise RuntimeError('Ran out of geoms. maxgeom: %d' % self._scn.maxgeom)

        g = mujoco.mjtGeom(self._scn.geoms + self._scn.ngeom)

        # default values.
        g.dataid = -1
        g.objtype = mujoco.mjtObj.OBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.CAT_DECOR
        g.texid = -1
        g.texuniform = 0
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mujoco.mjtGeom.GEOM_BOX
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3).flatten()
        g.rgba[:] = np.ones(4)

        for key, value in marker_params.items():
            if isinstance(value, (int, float)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                assert key == "label", "Only label is a string in mjvGeom."
                if value == None:
                    g.label[0] = 0
                else:
                    g.label = value
                    # strncpy(g.label, value.encode(), 100)
            elif hasattr(g, key):
                raise ValueError("mjvGeom has attr {} but type {} is invalid".format(key, type(value)))
            else:
                raise ValueError("mjvGeom doesn't have field %s" % key)

        self._scn.ngeom += 1

    def __del__(self):
        self._con.free()
        mujoco.mjv_freeScene(self._scn)


class MjRenderContextWindow(MjRenderContext):

    def __init__(self, sim):
        super().__init__(sim, offscreen=False)
        self.render_swap_callback = None
        assert isinstance(self.opengl_context, GlfwContext), "Only GlfwContext supported for windowed rendering"

    @property
    def window(self):
        return self.opengl_context.window

    def render(self):
        if self.window is None or glfw.window_should_close(self.window):
            return
        # else...
        glfw.make_context_current(self.window)
        super().render(*glfw.get_framebuffer_size(self.window))
        if self.render_swap_callback is not None:
            self.render_swap_callback()
        glfw.swap_buffers(self.window)


class MjViewerBasic(MjRenderContextWindow):
    """
    A simple display GUI showing the scene of an :class:`.MjSim` with a mouse-movable camera.

    :class:`.MjViewer` extends this class to provide more sophisticated playback and interaction controls.

    Parameters
    ----------
    sim : :class:`.MjSim`
        The simulator to display.
    """

    def __init__(self, sim):
        super().__init__(sim)

        self._gui_lock = Lock()
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0

        framebuffer_width, _ = glfw.get_framebuffer_size(self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = framebuffer_width * 1.0 / window_width

        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(
            self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self.key_callback)

    def render(self):
        """
        Render the current simulation state to the screen or off-screen buffer.
        Call this in your main loop.
        """
        if self.window is None:
            return
        elif glfw.window_should_close(self.window):
            glfw.terminate()
            sys.exit(0)

        with self._gui_lock:
            super().render()

        glfw.poll_events()

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.RELEASE and key == glfw.KEY_ESCAPE:
            print("Pressed ESC")
            print("Quitting.")
            glfw.terminate()
            sys.exit(0)

    def _cursor_pos_callback(self, window, xpos, ypos):
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        # Determine whether to move, zoom or rotate view
        mod_shift = (
                glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if self._button_right_pressed:
            action = mujoco.mjtGridPos.MOUSE_MOVE_H if mod_shift else mujoco.mjtGridPos.MOUSE_MOVE_V
        elif self._button_left_pressed:
            action = mujoco.mjtGridPos.MOUSE_ROTATE_H if mod_shift else mujoco.mjtGridPos.MOUSE_ROTATE_V
        else:
            action = mujoco.mjtGridPos.MOUSE_ZOOM

        # Determine
        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        with self._gui_lock:
            self.move_camera(action, dx / height, dy / height)

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window, button, act, mods):
        self._button_left_pressed = (
                glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self._button_right_pressed = (
                glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

    def _scroll_callback(self, window, x_offset, y_offset):
        with self._gui_lock:
            self.move_camera(mujoco.mjtGridPos.MOUSE_ZOOM, 0, -0.05 * y_offset)


class MjViewer(MjViewerBasic):
    """
    Extends :class:`.MjViewerBasic` to add video recording, interactive time and interaction controls.

    The key bindings are as follows:

    - TAB: Switch between MuJoCo cameras.
    - H: Toggle hiding all GUI components.
    - SPACE: Pause/unpause the simulation.
    - RIGHT: Advance simulation by one step.
    - V: Start/stop video recording.
    - T: Capture screenshot.
    - I: Drop into ``ipdb`` debugger.
    - S/F: Decrease/Increase simulation playback speed.
    - C: Toggle visualization of contact forces (off by default).
    - D: Enable/disable frame skipping when rendering lags behind real time.
    - R: Toggle transparency of geoms.
    - M: Toggle display of mocap bodies.
    - 0-4: Toggle display of geomgroups

    Parameters
    ----------
    sim : :class:`.MjSim`
        The simulator to display.
    """

    def __init__(self, sim):
        super().__init__(sim)

        self._ncam = sim.model.ncam
        self._paused = False  # is viewer paused.
        # should we advance viewer just by one step.
        self._advance_by_one_step = False

        # Vars for recording video
        self._record_video = False
        self._video_queue = Queue()
        self._video_idx = 0
        self._video_path = "/tmp/video_%07d.mp4"

        # vars for capturing screen
        self._image_idx = 0
        self._image_path = "/tmp/frame_%07d.png"

        # run_speed = x1, means running real time, x2 means fast-forward times
        # two.
        self._run_speed = 1.0
        self._loop_count = 0
        self._render_every_frame = False

        self._show_mocap = True  # Show / hide mocap bodies.
        self._transparent = False  # Make everything transparent.

        # this variable is estamated as a running average.
        self._time_per_render = 1 / 60.0
        self._hide_overlay = False  # hide the entire overlay.
        self._user_overlay = {}

    def render(self):
        """
        Render the current simulation state to the screen or off-screen buffer.
        Call this in your main loop.
        """

        def render_inner_loop(self):
            render_start = time.time()

            self._overlay.clear()
            if not self._hide_overlay:
                for k, v in self._user_overlay.items():
                    self._overlay[k] = copy.deepcopy(v)
                self._create_full_overlay()
            super().render()
            if self._record_video:
                frame = self._read_pixels_as_in_window()
                self._video_queue.put(frame)
            else:
                self._time_per_render = 0.9 * self._time_per_render + \
                                        0.1 * (time.time() - render_start)

        self._user_overlay = copy.deepcopy(self._overlay)
        # Render the same frame if paused.
        if self._paused:
            while self._paused:
                render_inner_loop(self)
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            # inner_loop runs "_loop_count" times in expectation (where "_loop_count" is a float).
            # Therefore, frames are displayed in the real-time.
            self._loop_count += self.sim.model.opt.timestep * self.sim.nsubsteps / \
                                (self._time_per_render * self._run_speed)
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                render_inner_loop(self)
                self._loop_count -= 1
        # Markers and overlay are regenerated in every pass.
        self._markers[:] = []
        self._overlay.clear()

    def _read_pixels_as_in_window(self, resolution=None):
        # Reads pixels with markers and overlay from the same camera as screen.
        if resolution is None:
            resolution = glfw.get_framebuffer_size(self.sim._render_context_window.window)

        resolution = np.array(resolution)
        resolution = resolution * min(1000 / np.min(resolution), 1)
        resolution = resolution.astype(np.int32)
        resolution -= resolution % 16
        if self.sim._render_context_offscreen is None:
            self.sim.render(resolution[0], resolution[1])
        offscreen_ctx = self.sim._render_context_offscreen
        window_ctx = self.sim._render_context_window
        # Save markers and overlay from offscreen.
        saved = [copy.deepcopy(offscreen_ctx._markers),
                 copy.deepcopy(offscreen_ctx._overlay),
                 rec_copy(offscreen_ctx.cam)]
        # Copy markers and overlay from window.
        offscreen_ctx._markers[:] = window_ctx._markers[:]
        offscreen_ctx._overlay.clear()
        offscreen_ctx._overlay.update(window_ctx._overlay)
        rec_assign(offscreen_ctx.cam, rec_copy(window_ctx.cam))

        img = self.sim.render(*resolution)
        img = img[::-1, :, :]  # Rendered images are upside-down.
        # Restore markers and overlay to offscreen.
        offscreen_ctx._markers[:] = saved[0][:]
        offscreen_ctx._overlay.clear()
        offscreen_ctx._overlay.update(saved[1])
        rec_assign(offscreen_ctx.cam, saved[2])
        return img

    def _create_full_overlay(self):
        if self._render_every_frame:
            self.add_overlay(mujoco.mjtGridPos.GRID_TOPLEFT, "", "")
        else:
            self.add_overlay(mujoco.mjtGridPos.GRID_TOPLEFT, "Run speed = %.3f x real time" %
                             self._run_speed, "[S]lower, [F]aster")
        self.add_overlay(
            mujoco.mjtGridPos.GRID_TOPLEFT, "Ren[d]er every frame", "Off" if self._render_every_frame else "On")
        self.add_overlay(mujoco.mjtGridPos.GRID_TOPLEFT, "Switch camera (#cams = %d)" % (self._ncam + 1),
                         "[Tab] (camera ID = %d)" % self.cam.fixedcamid)
        self.add_overlay(mujoco.mjtGridPos.GRID_TOPLEFT, "[C]ontact forces", "Off" if self.vopt.flags[
                                                                                          10] == 1 else "On")
        self.add_overlay(
            mujoco.mjtGridPos.GRID_TOPLEFT, "Referenc[e] frames", "Off" if self.vopt.frame == 1 else "On")
        self.add_overlay(mujoco.mjtGridPos.GRID_TOPLEFT,
                         "T[r]ansparent", "On" if self._transparent else "Off")
        self.add_overlay(
            mujoco.mjtGridPos.GRID_TOPLEFT, "Display [M]ocap bodies", "On" if self._show_mocap else "Off")
        if self._paused is not None:
            if not self._paused:
                self.add_overlay(mujoco.mjtGridPos.GRID_TOPLEFT, "Stop", "[Space]")
            else:
                self.add_overlay(mujoco.mjtGridPos.GRID_TOPLEFT, "Start", "[Space]")
            self.add_overlay(mujoco.mjtGridPos.GRID_TOPLEFT,
                             "Advance simulation by one step", "[right arrow]")
        self.add_overlay(mujoco.mjtGridPos.GRID_TOPLEFT, "[H]ide Menu", "")
        if self._record_video:
            ndots = int(7 * (time.time() % 1))
            dots = ("." * ndots) + (" " * (6 - ndots))
            self.add_overlay(mujoco.mjtGridPos.GRID_TOPLEFT,
                             "Record [V]ideo (On) " + dots, "")
        else:
            self.add_overlay(mujoco.mjtGridPos.GRID_TOPLEFT, "Record [V]ideo (Off) ", "")
        if self._video_idx > 0:
            fname = self._video_path % (self._video_idx - 1)
            self.add_overlay(mujoco.mjtGridPos.GRID_TOPLEFT, "   saved as %s" % fname, "")

        self.add_overlay(mujoco.mjtGridPos.GRID_TOPLEFT, "Cap[t]ure frame", "")
        if self._image_idx > 0:
            fname = self._image_path % (self._image_idx - 1)
            self.add_overlay(mujoco.mjtGridPos.GRID_TOPLEFT, "   saved as %s" % fname, "")
        self.add_overlay(mujoco.mjtGridPos.GRID_TOPLEFT, "Start [i]pdb", "")
        if self._record_video:
            extra = " (while video is not recorded)"
        else:
            extra = ""
        self.add_overlay(mujoco.mjtGridPos.GRID_BOTTOMLEFT, "FPS", "%d%s" %
                         (1 / self._time_per_render, extra))
        self.add_overlay(mujoco.mjtGridPos.GRID_BOTTOMLEFT, "Solver iterations", str(
            self.sim.data.solver_iter + 1))
        step = round(self.sim.data.time / self.sim.model.opt.timestep)
        self.add_overlay(mujoco.mjtGridPos.GRID_BOTTOMRIGHT, "Step", str(step))
        self.add_overlay(mujoco.mjtGridPos.GRID_BOTTOMRIGHT, "timestep", "%.5f" % self.sim.model.opt.timestep)
        self.add_overlay(mujoco.mjtGridPos.GRID_BOTTOMRIGHT, "n_substeps", str(self.sim.nsubsteps))
        self.add_overlay(mujoco.mjtGridPos.GRID_TOPLEFT, "Toggle geomgroup visibility", "0-4")

    def key_callback(self, window, key, scancode, action, mods):
        if action != glfw.RELEASE:
            return
        elif key == glfw.KEY_TAB:  # Switches cameras.
            self.cam.fixedcamid += 1
            self.cam.type = mujoco.mjtGridPos.CAMERA_FIXED
            if self.cam.fixedcamid >= self._ncam:
                self.cam.fixedcamid = -1
                self.cam.type = mujoco.mjtGridPos.CAMERA_FREE
        elif key == glfw.KEY_H:  # hides all overlay.
            self._hide_overlay = not self._hide_overlay
        elif key == glfw.KEY_SPACE and self._paused is not None:  # stops simulation.
            self._paused = not self._paused
        # Advances simulation by one step.
        elif key == glfw.KEY_RIGHT and self._paused is not None:
            self._advance_by_one_step = True
            self._paused = True
        elif key == glfw.KEY_V or \
                (
                        key == glfw.KEY_ESCAPE and self._record_video):  # Records video. Trigers with V or if in progress by ESC.
            self._record_video = not self._record_video
            if self._record_video:
                fps = (1 / self._time_per_render)
                self._video_process = Process(target=save_video,
                                              args=(self._video_queue, self._video_path % self._video_idx, fps))
                self._video_process.start()
            if not self._record_video:
                self._video_queue.put(None)
                self._video_process.join()
                self._video_idx += 1
        elif key == glfw.KEY_T:  # capture screenshot
            img = self._read_pixels_as_in_window()
            imageio.imwrite(self._image_path % self._image_idx, img)
            self._image_idx += 1
        elif key == glfw.KEY_I:  # drops in debugger.
            print('You can access the simulator by self.sim')
            import ipdb
            ipdb.set_trace()
        elif key == glfw.KEY_S:  # Slows down simulation.
            self._run_speed /= 2.0
        elif key == glfw.KEY_F:  # Speeds up simulation.
            self._run_speed *= 2.0
        elif key == glfw.KEY_C:  # Displays contact forces.
            vopt = self.vopt
            vopt.flags[10] = vopt.flags[11] = not vopt.flags[10]
        elif key == glfw.KEY_D:  # turn off / turn on rendering every frame.
            self._render_every_frame = not self._render_every_frame
        elif key == glfw.KEY_E:
            vopt = self.vopt
            vopt.frame = 1 - vopt.frame
        elif key == glfw.KEY_R:  # makes everything little bit transparent.
            self._transparent = not self._transparent
            if self._transparent:
                self.sim.model.geom_rgba[:, 3] /= 5.0
            else:
                self.sim.model.geom_rgba[:, 3] *= 5.0
        elif key == glfw.KEY_M:  # Shows / hides mocap bodies
            self._show_mocap = not self._show_mocap
            for body_idx1, val in enumerate(self.sim.model.body_mocapid):
                if val != -1:
                    for geom_idx, body_idx2 in enumerate(self.sim.model.geom_bodyid):
                        if body_idx1 == body_idx2:
                            if not self._show_mocap:
                                # Store transparency for later to show it.
                                self.sim.extras[
                                    geom_idx] = self.sim.model.geom_rgba[geom_idx, 3]
                                self.sim.model.geom_rgba[geom_idx, 3] = 0
                            else:
                                self.sim.model.geom_rgba[
                                    geom_idx, 3] = self.sim.extras[geom_idx]
        elif key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
            self.vopt.geomgroup[key - glfw.KEY_0] ^= 1
        super().key_callback(window, key, scancode, action, mods)


# Separate Process to save video. This way visualization is
# less slowed down.


def save_video(queue, filename, fps):
    writer = imageio.get_writer(filename, fps=fps)
    while True:
        frame = queue.get()
        if frame is None:
            break
        writer.append_data(frame)
    writer.close()
