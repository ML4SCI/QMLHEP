import importlib


def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'qgnn_hep.models.GraphConvNet'."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_
