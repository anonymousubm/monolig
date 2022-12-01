import importlib

def import_class(class_identifier):
    module_component, class_component = class_identifier.rsplit(".", 1)
    return getattr(importlib.import_module(module_component), class_component)


def initialize_class(class_definition, *args, **kwargs):
    if isinstance(class_definition, dict):
        class_type = class_definition["type"]
        kwargs.update(
            {name: value for name, value in class_definition.items() if name != "type"}
        )
    else:
        class_type = class_definition
    return import_class(class_type)(*args, **kwargs)
