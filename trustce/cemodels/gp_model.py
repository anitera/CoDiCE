from .model_interface import ModelInterface
import json
from trustce.ceinstance import CEInstance

class GeneticProgrammingModel(ModelInterface):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.model_type = model_config.get('model_type')
        self.model_name = model_config.get('name')
        self.model_state = model_config.get('state')
        self.model_path = model_config.get('path')

        if self.model_state == "pretrained":
            self.models = self.load_model(self.model_path)
            if model_config.get('gp_params') is not None:
                model_number = model_config.get('gp_params').get('model_number')
                # Depending if user enumerates models from 1 or 0
                self.model = self.models[model_number-1]
                print("First model is ", self.model)
        else: self.model = self.train(model_config)

    def train(self, model_config):
        # Implement the train method for sklearn model
        raise NotImplementedError

    def fit(self, X, y):
        # Implement the fit method for genetic programming model
        raise NotImplementedError

    def predict(self, X: CEInstance):
        # Evaluate the model expression
        X_dict = X.get_values_dict()
        return self.evaluate_expression(self.model, X_dict)
    
    def predict_instance(self, x: CEInstance):
        # Evaluate the model expression
        return self.evaluate_expression(self.model, x.get_values_dict())

    def evaluate_expression(self, expr, X):
        # Define operations
        operations = {
            'add': lambda a, b: a + b,
            'sub': lambda a, b: a - b,
            'mult': lambda a, b: a * b,
            'div': lambda a, b: a / b if b != 0 else float('inf'),
            'sign': lambda a: -1 if a < 0 else (1 if a > 0 else 0),
            # Add other operations as needed
        }

        # Evaluate the expression
        if isinstance(expr, tuple):
            function, args = expr[0], expr[1]
            evaluated_args = [self.evaluate_expression(arg, X) for arg in args]
            return operations[function](*evaluated_args)
        elif isinstance(expr, str) and expr.isidentifier():
            return X.get(expr, 0)  # Get the value from X or default to 0
        else:
            return float(expr)  # Convert numeric strings to float

    def evaluate(self, X, y):
        # Implement the evaluate method for genetic programming model
        pass

    def load_model(self, filepath):
        # Let's assume that the file is structured in the same way as trust-models.json 
        # and the indicator is name
        # Implement the load_model method for genetic programming model
        with open(filepath, 'rb') as f:
            self.model = json.load(f)
        # Extract the model based on the name
        gp_model = self.model[self.model_name]
        if gp_model is None:
            raise Exception("Model {} not found in {}".format(self.model_name, filepath))
        parsed_expressions = self.parse_expressions(gp_model)
        print("Parsed expressions are ", parsed_expressions)
        return parsed_expressions
    
    def parse_expressions(self, expressions):
        """Parsing multiple gp models from the following format:
        "Cart-Pole": [
        "add(div(Req1_Timeout, Restaurant_X), div(Driver_X, Req2_Y))",
        "add(add(sign(div(Driver_X, Restaurant_Y)), sub(mult(Driver_X, Driver_Used_Cap), mult(Restaurant_X, Driver_Full_Cap))), mult(add(div(Driver_Y, Req1_Res_Map), sign(Req2_Y)), add(Req2_Y, Req2_Res_Map)))",
        "mult(sign(div(Req1_Promise, Req1_Status)), mult(sign(sign(Req1_Timeout)), add(sub(Driver_Full_Cap, Req2_Y), div(Req2_Y, Req1_Status))))"
        ],
        """
        def parse_recursive(expr):
            # Base case: if the expression is a variable or a value
            if expr.isnumeric() or expr.isidentifier():
                return expr

            # Find the first opening parenthesis
            start = expr.find('(')
            if start == -1:
                return expr

            # Extract the function name and the argument list
            function = expr[:start].strip()
            args_list = expr[start + 1:-1]

            # Split the arguments and recursively parse each
            args = self.split_args(args_list)
            parsed_args = [parse_recursive(arg.strip()) for arg in args]

            return (function, parsed_args)

        parsed_models = []
        for expression in expressions:
            parsed_model = parse_recursive(expression)
            parsed_models.append(parsed_model)

        return parsed_models
    
    def split_args(self, args):
            """Split arguments of a function, considering nested structures."""
            args_list = []
            bracket_level = 0
            current_arg = ""

            for char in args:
                if char == '(':
                    bracket_level += 1
                elif char == ')':
                    bracket_level -= 1

                if char == ',' and bracket_level == 0:
                    args_list.append(current_arg)
                    current_arg = ""
                else:
                    current_arg += char

            if current_arg:
                args_list.append(current_arg)

            return args_list


    def sanity_check(self):
        return super().sanity_check()