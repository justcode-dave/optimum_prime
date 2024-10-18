"""
base.py

This module defines the base configuration class for exporting models in different formats. It establishes the foundation
for more specific export configurations to be built on top of this base. The class `ExportConfig` is an abstract base 
class (ABC) that other export configurations must inherit and extend.

Classes:
    ExportConfig: An abstract base class that serves as a blueprint for specific export configurations.
"""

# Import the Abstract Base Class (ABC) from the abc module to define abstract classes.
from abc import ABC

# Define the base class for all export configurations.
class ExportConfig(ABC):
    """
    ExportConfig serves as the base class for export configurations.

    This abstract base class (ABC) is meant to be inherited by other export classes that deal with specific
    formats such as ONNX, TFLite, etc. It provides a foundation for defining export-related settings and
    functionalities for models in Hugging Face's Optimum framework.
    
    Since this class does not implement any methods, it acts as a placeholder to enforce the implementation
    of methods in derived classes.

    Attributes:
        (No attributes or methods are defined at this level. This class is used for structure and enforcement.)
    """
    pass  # The class currently does not define any methods or attributes.
