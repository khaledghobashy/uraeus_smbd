import json

from .encoder import SMBDEncoder
from smbd.symbolic.systems.configuration_classes import abstract_configuration

class configuration(object):

    def __init__(self, sym_config):

        # Checking whether the given instance is of the expected type
        if not isinstance(sym_config, abstract_configuration):
            raise ValueError('"sym_config" should be an instance of "abstract_configuration" class!')

        # Storing the given symbolic configuration instance in "config" attribute
        self.config = sym_config
        self.name = self.config.name
        
        # Extracting the needed attributes from the "config" attribute.

        # Creating a dict that stores the inputs' equalities as input_name : input_data.
        self.inputs  = {str(expr.lhs): expr.rhs for expr in self.config.input_equalities}

        self.data = {"inputs": self.inputs}

    def dump_JSON(self):
        json_text = json.dumps(self.data, cls=SMBDEncoder, indent=4)
        print(json_text)
