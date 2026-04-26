from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union
import os
import warnings
from . import pynokhwa
import sys

try:
    import numpy as np
except ImportError:
    pass

try:
    from PIL import Image
except ImportError:
    pass


@dataclass
class CameraInfo:
    """
    Describes a connected camera.
    """

    index: int
    name: str
    description: str
    misc: str
    unique_id: str = ""
    id_stable: bool = False

    def can_open(self):
        """
        Check if this camera can be opened.
        """
        return pynokhwa.check_can_use(self.index)


class FrameFormat(Enum):
    MJPEG = "mjpeg"
    YUYV = "yuyv"
    GRAY = "gray"
    NV12 = "nv12"
    RAWRGB = "rawrgb"


class CameraFormat:
    def __init__(self, cam_format: pynokhwa.CamFormat):
        self._fmt = cam_format

    @property
    def width(self) -> int:
        return self._fmt.width

    @property
    def height(self) -> int:
        return self._fmt.height

    @property
    def frame_rate(self) -> int:
        return self._fmt.frame_rate

    @property
    def frame_format(self) -> FrameFormat:
        return FrameFormat(self._fmt.format)

    def __str__(self) -> str:
        return (
            f"{self.frame_format.value} {self.width}x{self.height}@{self.frame_rate}fps"
        )


class CameraFormatOptions(list):
    """
    A list of camera formats with additional methods to choose the most fitting camera format.
    prefer* functions either return only fitting formats or everything, in case there are no fitting formats.
    """

    def prefer(self, func):
        """
        Prefer formats for which func is true.
        """
        options = CameraFormatOptions(filter(func, self))
        if not options:
            return self
        return options

    def _prefer_range(self, val_getter, min_val=None, max_val=None):
        return self.prefer(
            lambda x: (min_val is None or val_getter(x) >= min_val)
            and (max_val is None or val_getter(x) <= max_val)
        )

    def prefer_fps_range(self, min_fps: int = None, max_fps: int = None):
        """
        Prefer formats with min_fps <= frame_rate <= max_fps.
        """
        return self._prefer_range(lambda x: x.frame_rate, min_fps, max_fps)

    def prefer_width_range(self, min_width: int = None, max_width: int = None):
        """
        Prefer formats with min_width <= width <= max_width.
        """
        return self._prefer_range(lambda x: x.width, min_width, max_width)

    def prefer_height_range(self, min_heigth: int = None, max_height: int = None):
        """
        Prefer formats with min_heigth <= height <= max_height.
        """
        return self._prefer_range(lambda x: x.height, min_heigth, max_height)

    def prefer_aspect_ratio(self, width_by_height: float):
        """
        Prefer formats with width / height == width_by_height
        """
        return self.prefer(lambda x: abs(x.width / x.height - width_by_height) < 1e-6)

    def prefer_sides_ratio(self, width_by_height: float):
        warnings.warn(
            "prefer_sides_ratio has been renamed to prefer_aspect_ratio",
            DeprecationWarning,
        )
        return self.prefer_aspect_ratio(width_by_height)

    def prefer_frame_format(self, fmt: FrameFormat):
        return self.prefer(lambda x: x.frame_format is fmt)

    def resolve(self, key=lambda x: x.width) -> CameraFormat:
        """
        Returns one of contained formats.
        """
        return max(self, key=key)

    def find_highest_resolution(formats: List[CameraFormat]) -> CameraFormat:
        """
        Pick the format with the highest resolution.
        If multiple formats have the same resolution, pick the one with the highest framerate.
        """
        return max(formats, key=lambda f: (f.width, f.height, f.frame_rate))

    def find_highest_framerate(formats: List[CameraFormat]) -> CameraFormat:
        """
        Pick the format with the highest framerate.
        If multiple formats have the same framerate, pick the one with the highest resolution.
        """
        return max(formats, key=lambda f: (f.frame_rate, f.width, f.height))

    def resolve_default(self) -> CameraFormat:
        """
        Like resolve, but applies some default preferences.
        Used when camera is opened automatically.
        """
        return (
            self.prefer_fps_range(25, 60)
            .prefer_sides_ratio(4 / 3)
            .prefer_frame_format(FrameFormat.MJPEG)
            .resolve()
        )


class CameraControl:
    def __init__(self, control):
        self._control = control

    def __str__(self):
        return f"{self.value_range} : {self.is_active}"

    @property
    def value_range(self) -> range:
        """
        Return a range of values that can be passed to set_value.
        """
        start, stop, step = self._control.value_range()
        if step > 0:
            return range(start, stop, step)
        return range(start, stop)

    @property
    def is_active(self) -> bool:
        return self._control.is_active()

    def set_active(self, value: bool):
        self._control.set_active(value)

    def set_value(self, value: Union[int, None]):
        """
        Set a value for this control.
        *value* in self.value_range should be true.
        """
        if value is not None:
            assert value in self.value_range
        self._control.set_value(value)

    def set_fraction(self, fraction: float):
        """
        Set a value for this control.
        0 <= *fraction* <= 1 should be true.
        """
        assert 0 <= fraction <= 1
        ind = round(fraction * (len(self.value_range) - 1))
        self.set_value(self.value_range[ind])


class Camera:
    def __init__(self, info: Union[CameraInfo, int, str], suggested_fps: int = 25):
        """
        Open a camera corresponding to the info object.
        Will try to use maximum possible resolution with a frame rate of at least *suggested_fps*
        """
        if isinstance(info, CameraInfo):
            self.info = info
            camera_id = self._open_token_for_info(info)
        else:
            self.info = None
            camera_id = self._open_token_for_id(info)
            if camera_id is None:
                raise ValueError(f"Could not resolve camera unique_id: {info}")
        self._initialized = False
        self._cam = pynokhwa.Camera(camera_id)

    @staticmethod
    def _open_token_for_info(info: CameraInfo) -> Union[int, str]:
        unique_id = info.unique_id or info.misc
        if not unique_id:
            return info.index
        token = Camera._open_token_for_id(unique_id)
        if token is None:
            return info.index
        return token

    @staticmethod
    def _open_token_for_id(camera_id: Union[int, str]) -> Union[int, str, None]:
        if isinstance(camera_id, int):
            return camera_id

        if camera_id.startswith("index:"):
            try:
                return int(camera_id.removeprefix("index:"))
            except ValueError:
                return None

        if camera_id.isdigit():
            return int(camera_id)

        if sys.platform.startswith("linux"):
            index = Camera._linux_unique_id_to_index(camera_id)
            if index is not None:
                return index
            return None

        return camera_id

    @staticmethod
    def _linux_unique_id_to_index(unique_id: str) -> Union[int, None]:
        path = os.path.realpath(unique_id)
        name = os.path.basename(path)
        if not name.startswith("video"):
            return None
        try:
            return int(name.removeprefix("video"))
        except ValueError:
            return None

    def get_format_options(self) -> CameraFormatOptions:
        """
        Returns a list of supported CameraFormat objects.
        """
        return CameraFormatOptions(map(CameraFormat, self._cam.get_formats()))

    def get_controls(self) -> Dict[str, CameraControl]:
        """
        [Experimental]
        Get a list of supported camera controls
        """
        return {k: CameraControl(v) for k, v in self._cam.get_controls()}

    def open(self, fmt: CameraFormat = None):
        """
        Select a format and open the camera.
        Called automatically if needed from poll_frame_* method family.
        """
        if self._initialized:
            raise RuntimeError("Can only open once")
        if fmt is None:
            fmt = self.get_format_options().resolve_default()
        self._cam.open(fmt._fmt)
        self._initialized = True

    def close(self):
        """
        Close the camera
        """
        self._cam.close()
        self._initialized = False

    def _poll_raw(self) -> Union[tuple[int, int, bytes, int], None]:
        """Shared implementation: returns (w, h, raw_rgb_bytes, seq) or None."""
        if not self._initialized:
            self.open()
        self._cam.check_err()
        return self._cam.poll_frame_with_seq()

    def poll_frame_raw(self) -> Union[tuple[int, int, bytes], None]:
        """
        Get a frame from the camera. Returns width, height, and array of raw rgb values.
        Guaranteed to never block, but may return None if no frames were received from camera yet.
        """
        result = self._poll_raw()
        if result is None:
            return None
        w, h, data, _seq = result
        return (w, h, data)

    def poll_frame_raw_with_seq(self) -> Union[tuple[int, int, bytes, int], None]:
        """
        Like poll_frame_raw(), but also returns the frame sequence number.
        The sequence increments only when the camera produces a genuinely new frame.
        Returns (width, height, raw_rgb_bytes, sequence) or None.
        """
        return self._poll_raw()

    @staticmethod
    def _raw_to_np(w, h, data):
        expected_size = w * h * 3
        if len(data) > expected_size:
            data = data[:expected_size]
        return np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))

    def poll_frame_np(self) -> Union["np.ndarray", None]:
        """
        Get a frame from the camera. Returns a numpy array.
        Guaranteed to never block, but may return None if no frames were received from camera yet.
        """
        result = self._poll_raw()
        if result is None:
            return None
        w, h, data, _seq = result
        return self._raw_to_np(w, h, data)

    def poll_frame_np_with_seq(self) -> Union[tuple["np.ndarray", int], None]:
        """
        Like poll_frame_np(), but also returns the frame sequence number.
        Returns (numpy_array, sequence) or None.
        """
        result = self._poll_raw()
        if result is None:
            return None
        w, h, data, seq = result
        return self._raw_to_np(w, h, data), seq

    def poll_frame_pil(self) -> Union["Image.Image", None]:
        """
        Get a frame from the camera. Returns a pillow image.
        Guaranteed to never block, but may return None if no frames were received from camera yet.
        """
        result = self._poll_raw()
        if result is None:
            return None
        w, h, data, _seq = result
        return Image.frombytes("RGB", (w, h), data)

    def _info(self):
        return self._cam.info()


def query(only_usable=True) -> list[CameraInfo]:
    """
    Returns a list of CameraInfo objects, one for every available camera.
    """
    result = map(lambda x: CameraInfo(*x), pynokhwa.query())
    if only_usable:
        result = filter(CameraInfo.can_open, result)
    return list(result)
