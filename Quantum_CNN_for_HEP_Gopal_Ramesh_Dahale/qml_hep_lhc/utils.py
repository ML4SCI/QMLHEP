from importlib import import_module
from argparse import Action


def _import_class(module_and_class_name: str) -> type:
    """
		Import class from a module, e.g. 'qml_hep_lhc.models.QNN'
		
		Args:
			module_and_class_name (str): str
		
		Returns:
			A class
		"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


class ParseAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = list(map(int, values.split()))
        setattr(namespace, self.dest, values)
