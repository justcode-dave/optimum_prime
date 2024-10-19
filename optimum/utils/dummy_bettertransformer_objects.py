"""
Dummy Objects for BetterTransformer Integration

This module defines dummy objects for the BetterTransformer integration in Promise Optimizer. 
These objects serve as placeholders when the required backends (such as specific versions of the 
Transformers library) are not available in the environment. This approach ensures that the code 
fails gracefully and provides informative errors when attempting to use functionalities that 
depend on unavailable dependencies.

Classes:
    - BarkSelfAttention: A dummy class for the `BarkSelfAttention` object that requires a specific 
      backend (`transformers_431`). The class raises an error if the necessary backend is not available.
"""

from .import_utils import DummyObject, requires_backends  # Importing utility functions for dummy objects


class BarkSelfAttention(metaclass=DummyObject):
    """
    Dummy implementation of the BarkSelfAttention module for BetterTransformer.

    This class is designed to act as a placeholder when the `transformers_431` backend 
    is not installed. It uses a custom metaclass `DummyObject` that helps raise an informative 
    error whenever an attempt is made to instantiate or use this class without the required backend.

    Attributes:
        _backends (List[str]): The list of required backends, in this case `transformers_431`.
    """
    _backends = ["transformers_431"]  # Specifies that the class requires `transformers_431` backend

    def __init__(self, *args, **kwargs):
        """
        Initializes the BarkSelfAttention dummy class.

        This method attempts to ensure that the required backends (`transformers_431`) 
        are installed before allowing the object to be instantiated. If the backends are 
        not present, it raises an appropriate error.

        Args:
            *args: Variable-length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        requires_backends(self, ["transformers_431"])  # Check for required backends during instantiation
