"""
Documentation Utilities for Promise Optimizer

This module contains utility functions for dynamically generating and updating docstrings for 
dataclasses and functions. These utilities help in automatically creating and formatting documentation, 
ensuring that the codebase is well-documented without requiring manual docstring updates for 
every change in attributes or function behavior.

Functions:
    - generate_doc_dataclass: A decorator that automatically generates docstrings for attributes in dataclasses.
    - add_dynamic_docstring: A decorator that dynamically adds or modifies a function's docstring with provided text 
      and dynamic elements.
"""

from dataclasses import fields  # Importing fields to introspect dataclass attributes


def generate_doc_dataclass(cls) -> str:
    """
    Class decorator that automatically generates documentation for dataclass attributes.

    This function generates a docstring for a dataclass by iterating over its fields and appending 
    information about each attribute, including its type and description, if provided.

    Args:
        cls (dataclass): The dataclass for which to generate the documentation.

    Returns:
        dataclass: The same dataclass, but with an updated `__doc__` string that includes attribute details.
    """
    doc = "\f\nAttributes:\n"
    for attribute in fields(cls):  # Loop through all fields in the dataclass
        doc += f"   {attribute.name}"  # Add attribute name to the docstring

        # Determine if the attribute is optional and display its type
        attribute_type = str(attribute.type)
        if attribute_type.startswith("typing.Optional"):
            optional = True
            type_display = attribute_type[attribute_type.find("[") + 1 : -1]
            type_display = type_display.split(".")[-1]  # Get only the base type
        else:
            optional = False
            if attribute_type.startswith("typing"):
                type_display = attribute_type.split(".")[-1]  # Extract the base type from typing hints
            else:
                type_display = attribute.type.__name__

        # Add type information to the docstring, indicating whether it's optional
        if optional:
            doc += f" (`{type_display}`, *optional*): "
        else:
            doc += f" (`{type_display}`): "

        # Append attribute description to the docstring
        doc += f"{attribute.metadata['description']}\n"
    
    # Append the generated docstring to the class' existing docstring (if any)
    cls.__doc__ = (cls.__doc__ if cls.__doc__ is not None else "") + "\n\n" + "".join(doc)
    return cls


def add_dynamic_docstring(
    *docstr,
    text,
    dynamic_elements,
):
    """
    Function decorator that dynamically adds or modifies a function's docstring.

    This utility function allows dynamic generation of docstrings, where predefined text can be combined 
    with dynamic elements (such as parameter names or other context-specific information). This is useful 
    when certain parts of a docstring are generated or updated programmatically.

    Args:
        *docstr: Additional docstring content to prepend to the function's docstring.
        text (str): The text template to dynamically insert into the function's docstring.
        dynamic_elements (dict): A dictionary of dynamic elements to format into the `text`.

    Returns:
        function: The decorated function with an updated docstring.
    """
    def docstring_decorator(fn):
        # Combine the original docstring with the additional content and dynamic elements
        func_doc = (fn.__doc__ or "") + "".join(docstr)
        fn.__doc__ = func_doc + text.format(**dynamic_elements)  # Format dynamic elements into the text
        return fn

    return docstring_decorator
