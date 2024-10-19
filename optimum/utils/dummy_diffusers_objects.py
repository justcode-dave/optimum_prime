"""
Dummy Objects for Diffusers Pipelines (First Half)

This module defines dummy classes for several diffusion-related pipelines used in the Promise Optimizer project. 
These dummy classes act as placeholders when the required backend (`diffusers`) is not available in the environment. 
By using these placeholders, the system ensures that an informative error is raised, guiding the user to install the 
necessary dependencies if they are not already present.

Classes:
    - ORTStableDiffusionPipeline
    - ORTStableDiffusionImg2ImgPipeline
    - ORTStableDiffusionInpaintPipeline
    - ORTStableDiffusionXLPipeline
    - ORTStableDiffusionXLImg2ImgPipeline
    - ORTLatentConsistencyModelPipeline

Each class checks for the availability of the `diffusers` backend before being instantiated or used.
"""

from .import_utils import DummyObject, requires_backends  # Import utilities for dummy objects and backend checks


class ORTStableDiffusionPipeline(metaclass=DummyObject):
    """
    Dummy class for `ORTStableDiffusionPipeline`, requiring the `diffusers` backend.

    This class raises an error if the `diffusers` package is not installed.
    """
    _backends = ["diffusers"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the ORTStableDiffusionPipeline dummy class.

        Args:
            *args: Variable-length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(self, ["diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Attempts to load a pretrained model using the `diffusers` backend.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(cls, ["diffusers"])


class ORTStableDiffusionImg2ImgPipeline(metaclass=DummyObject):
    """
    Dummy class for `ORTStableDiffusionImg2ImgPipeline`, requiring the `diffusers` backend.
    """
    _backends = ["diffusers"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the ORTStableDiffusionImg2ImgPipeline dummy class.

        Args:
            *args: Variable-length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(self, ["diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Attempts to load a pretrained model using the `diffusers` backend.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(cls, ["diffusers"])


class ORTStableDiffusionInpaintPipeline(metaclass=DummyObject):
    """
    Dummy class for `ORTStableDiffusionInpaintPipeline`, requiring the `diffusers` backend.

    This class raises an error if the `diffusers` package is not installed.
    """
    _backends = ["diffusers"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the ORTStableDiffusionInpaintPipeline dummy class.

        Args:
            *args: Variable-length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(self, ["diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Attempts to load a pretrained model using the `diffusers` backend.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(cls, ["diffusers"])


class ORTStableDiffusionXLPipeline(metaclass=DummyObject):
    """
    Dummy class for `ORTStableDiffusionXLPipeline`, requiring the `diffusers` backend.

    This class raises an error if the `diffusers` package is not installed.
    """
    _backends = ["diffusers"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the ORTStableDiffusionXLPipeline dummy class.

        Args:
            *args: Variable-length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(self, ["diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Attempts to load a pretrained model using the `diffusers` backend.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(cls, ["diffusers"])


class ORTStableDiffusionXLImg2ImgPipeline(metaclass=DummyObject):
    """
    Dummy class for `ORTStableDiffusionXLImg2ImgPipeline`, requiring the `diffusers` backend.

    This class raises an error if the `diffusers` package is not installed.
    """
    _backends = ["diffusers"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the ORTStableDiffusionXLImg2ImgPipeline dummy class.

        Args:
            *args: Variable-length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(self, ["diffusers"])


@classmethod
def from_pretrained(cls, *args, **kwargs):
    """
    Attempts to load a pretrained model using the `diffusers` backend.

    Raises:
        ImportError: If the `diffusers` backend is not available.
    """
    requires_backends(cls, ["diffusers"])


class ORTLatentConsistencyModelPipeline(metaclass=DummyObject):
    """
    Dummy class for `ORTLatentConsistencyModelPipeline`, requiring the `diffusers` backend.

    This class raises an error if the `diffusers` package is not installed.
    """
    _backends = ["diffusers"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the ORTLatentConsistencyModelPipeline dummy class.

        Args:
            *args: Variable-length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(self, ["diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Attempts to load a pretrained model using the `diffusers` backend.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(cls, ["diffusers"])


class ORTDiffusionPipeline(metaclass=DummyObject):
    """
    Dummy class for `ORTDiffusionPipeline`, requiring the `diffusers` backend.

    This class raises an error if the `diffusers` package is not installed.
    """
    _backends = ["diffusers"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the ORTDiffusionPipeline dummy class.

        Args:
            *args: Variable-length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(self, ["diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Attempts to load a pretrained model using the `diffusers` backend.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(cls, ["diffusers"])


class ORTPipelineForText2Image(metaclass=DummyObject):
    """
    Dummy class for `ORTPipelineForText2Image`, requiring the `diffusers` backend.

    This class raises an error if the `diffusers` package is not installed.
    """
    _backends = ["diffusers"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the ORTPipelineForText2Image dummy class.

        Args:
            *args: Variable-length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(self, ["diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Attempts to load a pretrained model using the `diffusers` backend.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(cls, ["diffusers"])


class ORTPipelineForImage2Image(metaclass=DummyObject):
    """
    Dummy class for `ORTPipelineForImage2Image`, requiring the `diffusers` backend.

    This class raises an error if the `diffusers` package is not installed.
    """
    _backends = ["diffusers"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the ORTPipelineForImage2Image dummy class.

        Args:
            *args: Variable-length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(self, ["diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Attempts to load a pretrained model using the `diffusers` backend.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(cls, ["diffusers"])


class ORTPipelineForInpainting(metaclass=DummyObject):
    """
    Dummy class for `ORTPipelineForInpainting`, requiring the `diffusers` backend.

    This class raises an error if the `diffusers` package is not installed.
    """
    _backends = ["diffusers"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the ORTPipelineForInpainting dummy class.

        Args:
            *args: Variable-length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(self, ["diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Attempts to load a pretrained model using the `diffusers` backend.

        Raises:
            ImportError: If the `diffusers` backend is not available.
        """
        requires_backends(cls, ["diffusers"])
