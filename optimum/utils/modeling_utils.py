"""
modeling_utils.py

This module contains utility functions for manipulating and working with model attributes, specifically in a recursive
manner. It provides helper functions to recursively get and set attributes of a model, useful when working with 
nested structures in machine learning models.

Key Functions:
--------------
- `recurse_getattr(obj, attr: str)`:
    Recursively retrieves a nested attribute from an object using a dot-separated string.
    
- `recurse_setattr(module, name, value)`:
    Recursively sets a value to a nested attribute of a module using a dot-separated string.

Constants:
----------
- `MODEL_TO_PATCH_FOR_PAST`: 
    A set of model names where patching for past key values might be required. This is used to handle specific models 
    that need customization for managing past key-value states (like for decoder models).

This module is especially useful when working with complex model architectures that have deeply nested configurations 
or attributes.
"""

import functools  # Standard library module for higher-order functions like reduce.

# A set of model names that require patching to handle past key values in transformer-based architectures.
MODEL_TO_PATCH_FOR_PAST = {
    "bart",
    "blenderbot",
    "blenderbot-small",
    "bloom",
    "llama",
    "mistral",
    "mpt",
    "opt",
    "pegasus",
}


def recurse_getattr(obj, attr: str):
    """
    Recursively retrieves an attribute from an object, supporting nested attribute access via dot notation.

    Args:
        obj: 
            A class instance holding the attribute.
        attr (str): 
            The attribute to retrieve, which can be a nested attribute like 'attribute1.attribute2'.

    Returns:
        The value of the nested attribute if it exists.
    
    Example:
        ```
        class Example:
            def __init__(self):
                self.level1 = Level1()

        class Level1:
            def __init__(self):
                self.level2 = "some value"
        
        obj = Example()
        value = recurse_getattr(obj, "level1.level2")
        print(value)  # Output: "some value"
        ```
    """
    
    def _getattr(obj, attr):
        return getattr(obj, attr)  # Retrieves an attribute from the object.

    # Reduce applies _getattr iteratively across the split attribute names.
    return functools.reduce(_getattr, [obj] + attr.split("."))


def recurse_setattr(module, name, value):
    """
    Recursively sets an attribute's value in a module, supporting nested attribute access via dot notation.

    Args:
        module: 
            The module or object on which to set the attribute.
        name (str): 
            The attribute to set, which can be a nested attribute like 'module1.module2.attribute'.
        value: 
            The value to set the attribute to.

    Example:
        ```
        class Example:
            def __init__(self):
                self.level1 = Level1()

        class Level1:
            def __init__(self):
                self.level2 = None
        
        obj = Example()
        recurse_setattr(obj, "level1.level2", "new value")
        print(obj.level1.level2)  # Output: "new value"
        ```
    """
    if "." not in name:
        setattr(module, name, value)  # Directly set the attribute if there are no nested levels.
    else:
        name, rest = name.split(".", 1)  # Split the name at the first dot.
        recurse_setattr(getattr(module, name), rest, value)  # Recursively set the remaining part of the attribute.
