from active_learning.utils.class_utils import initialize_class


class CombinationLogic(object):
    def __init__(
        self,
        fn_definitions,
        logic_string,
    ):
        self.fn_string_to_class = {}
        for fn_string, fn_definition in fn_definitions.items():
            self.fn_string_to_class[fn_string] = initialize_class(fn_definition)
        self.logic_string = logic_string

    def eval_fn_string(self, logic_string, uncertainties):
        if len(logic_string) == 2 and logic_string[0] == "$":
            return uncertainties[int(logic_string[1])]
        fn_name = logic_string.split("(")[0]
        arguments = logic_string[len(fn_name) + 1 : -1]

        num_open_paranthesis = 0
        args_list = []
        current_arg_string = ""
        for chr in arguments:
            if chr == ",":
                if num_open_paranthesis == 0 and current_arg_string != "":
                    args_list.append(
                        self.eval_fn_string(current_arg_string, uncertainties)
                    )
                    current_arg_string = ""
                else:
                    current_arg_string += chr
            elif chr == "(":
                num_open_paranthesis += 1
                current_arg_string += chr
            elif chr == ")":
                num_open_paranthesis -= 1
                current_arg_string += chr
            else:
                current_arg_string += chr
        if current_arg_string != "":
            args_list.append(self.eval_fn_string(current_arg_string, uncertainties))
        return self.fn_string_to_class[fn_name].calculate(*args_list)

    def combine(self, uncertainties):
        return self.eval_fn_string(self.logic_string, uncertainties)
