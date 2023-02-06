from collections import defaultdict
from functools import partial, reduce
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np

from autoinf.inference.rules import (
    gen_nnsmith_rules,
    gen_requires_trees,
    gen_type_transfer_trees,
    judge_failure,
)
from autoinf.instrument.utils import (
    data_type_str,
    get_ret_list,
    hash_list_str,
    is_int_not_bool,
    numpy_random,
)


class AbsValue:
    def __init__(self, value):
        self.value = value

    def concretize(self, *args, **kwargs):
        return self.value

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return repr(self.value)

    def concrete_str(self, *args, **kwargs) -> str:
        return self.__str__()


class AbsInt(AbsValue):
    def concretize(self, symb_2_value: Dict[str, Any], *args, **kwargs):
        return symb_2_value[self.value]

    def concrete_str(self, symb_2_value: Dict[str, Any]) -> str:
        # s0=1
        if isinstance(self.value, str):
            return f"{self.value}={symb_2_value[self.value]}"
        else:
            return str(self.value)


class AbsTensor(AbsValue):
    def __init__(self, rank: int, shape: List[Union[str, int]], dtype: str):
        self.rank = rank
        self.shape = shape
        self.dtype = dtype

    @staticmethod
    def from_numpy(x: np.ndarray) -> "AbsTensor":
        return AbsTensor(x.ndim, list(x.shape), str(x.dtype))

    def concretize(
        self,
        symb_2_value: Dict[str, Any],
        tensor_from_numpy: Callable = lambda x: x,
        *args,
        **kwargs,
    ):
        shape = [symb_2_value[s] for s in self.shape]
        return tensor_from_numpy(numpy_random(shape, self.dtype))

    def concrete_shape(self, symb_2_value: Dict[str, Any]) -> List[int]:
        return [symb_2_value[s] for s in self.shape]

    def __str__(self) -> str:
        return f"AbsTensor<{self.rank}>({', '.join(self.shape)}, {self.dtype})"

    def concrete_str(self, symb_2_value: Dict[str, Any]) -> str:
        # AbsTensor<3>([s0=1, s1=2, s2=3], float32)
        shapes = [f"{s}={symb_2_value[s]}" for s in self.shape]
        return f"AbsTensor<{self.rank}>({', '.join(shapes)}, {self.dtype})"

    def __repr__(self) -> str:
        return str(self)


class OpInstance:
    def __init__(
        self,
        record: Dict[str, Any],
        int_policy: str = "fix",  # "symb", "fix_dim"
        masked_names: List[str] = None,
        dtype_class: Any = None,
    ):
        self.name: str = record["name"]
        self.int_policy = int_policy
        self.input_symb_2_value: Dict[str, Any] = {}
        self.output_symb_2_value: Dict[str, Any] = {}

        self.input_tensors: List[AbsTensor] = []
        self.output_tensors: List[AbsTensor] = []
        self.int_attrs: List[AbsInt] = []
        self.other_attrs: List[AbsValue] = []

        self.names: List[str] = []
        self.is_pos: List[bool] = []
        self.abs_values: List[AbsValue] = []
        self.input_types: List[str] = []

        """fill information by parsing the record"""
        record_args = record["args"]
        self.names = record_args["name"]
        self.is_pos = record_args["is_pos"]
        for i_arg, arg_value in enumerate(record_args["value"]):
            """fill self.input_types"""
            arg_name = self.names[i_arg]
            if masked_names and arg_name in masked_names:
                input_type_str = f"masked<{arg_name}>"
            else:
                input_type_str = data_type_str(
                    arg_value, self._keep_int_value(arg_name), dtype_class
                )
            self.input_types.append(input_type_str)
            """fill self.abs_values"""
            self.abs_values.append(self._add_input_arg(arg_value, arg_name))
        self.output_symb_2_value, self.output_tensors = self.output_info(
            record["outputs"]["value"]
        )

    def _keep_int_value(self, arg_name: str):
        return (
            self.int_policy == "fix"
            or self.int_policy == "fix_dim"
            and (
                "dim" in arg_name
                or "axis" in arg_name
                or
                # some hacks...
                (
                    self.name
                    in ["torch.movedim", "torch.Tensor.movedim", "torch.moveaxis"]
                    and arg_name in ["source", "destination"]
                )
                or (
                    self.name
                    in [
                        "torch.Tensor.diag",
                        "torch.diag",
                        "torch.diagonal",
                        "torch.Tensor.diagonal",
                        "torch.diagonal_copy",
                    ]
                    and arg_name in ["offset"]
                )
            )
        )

    def _add_input_arg(self, value, arg_name: str) -> Any:
        """
        1. symbolize it and update self.input_symb_2_value
        2. add AbsValue to value containers
        3. add code to self.abs_values
        Return: value to be added to self.abs_values
        """
        if isinstance(value, list):
            return [self._add_input_arg(v, arg_name) for v in value]
            # TODO model list with AbsValue to avoid manual recursion every time (like self._value_concrete_str)
        elif isinstance(value, np.ndarray):
            abs_tensor = AbsTensor.from_numpy(value)
            for i_s, s in enumerate(abs_tensor.shape):
                symb = f"s{len(self.input_symb_2_value)}"
                self.input_symb_2_value[symb] = s
                abs_tensor.shape[i_s] = symb
            self.input_tensors.append(abs_tensor)
            return abs_tensor
        elif not self._keep_int_value(arg_name) and is_int_not_bool(value):
            symb = f"s{len(self.input_symb_2_value)}"
            self.input_symb_2_value[symb] = int(value)
            abs_int = AbsInt(symb)
            self.int_attrs.append(abs_int)
            return abs_int
        else:
            abs_value = AbsValue(value)
            self.other_attrs.append(abs_value)
            return abs_value

    def _value_concrete_str(self, value, symb_2_value):
        if isinstance(value, list):
            return ", ".join([self._value_concrete_str(v, symb_2_value) for v in value])
        else:
            return value.concrete_str(symb_2_value)

    def __str__(self) -> str:
        if self.input_symb_2_value:
            inputs = [
                self._value_concrete_str(v, self.input_symb_2_value)
                for v in self.abs_values
            ]
        else:
            inputs = list(map(str, self.abs_values))
        inputs = [f"{name} = {value}" for name, value in zip(self.names, inputs)]
        if self.output_symb_2_value:
            outputs = [
                self._value_concrete_str(o, self.output_symb_2_value)
                for o in self.output_tensors
            ]
        else:
            outputs = list(map(str, self.output_tensors))
        name = getattr(self, "name_index", None)
        if not (name and isinstance(name, str)):
            name = self.name
        return f"OpInstance<{name}>( [{', '.join(inputs)}] -> [{', '.join(outputs)}] )"

    def concrete_str(
        self, input_symb_2_value: Dict[str, Any], output_symb_2_value: Dict[str, Any]
    ) -> str:
        inputs = [
            self._value_concrete_str(v, input_symb_2_value) for v in self.abs_values
        ]
        inputs = [f"{name} = {value}" for name, value in zip(self.names, inputs)]
        outputs = [
            self._value_concrete_str(o, output_symb_2_value)
            for o in self.output_tensors
        ]
        name = getattr(self, "name_index", None)
        if not (name and isinstance(name, str)):
            name = self.name
        return f"OpInstance<{name}>( [{', '.join(inputs)}] -> [{', '.join(outputs)}] )"

    def __repr__(self) -> str:
        return str(self)

    @property
    def signature_hash(self):
        out_rank_strs = [str(ts.rank) for ts in self.output_tensors]
        return hash_list_str([self.name] + self.input_types + out_rank_strs)

    @property
    def tensor_shape_hash(self):
        return hash_list_str(
            [str(t.concrete_shape(self.input_symb_2_value)) for t in self.input_tensors]
        )

    @property
    def record_hash(self):
        return hash_list_str(
            [str(t.concrete_shape(self.input_symb_2_value)) for t in self.input_tensors]
            + [str(i.concretize(self.input_symb_2_value)) for i in self.int_attrs]
        )

    @property
    def A(self) -> List[str]:
        # TODO: Only usable when int_policy=False
        return [
            n
            for n in self.input_symb_2_value
            if all([n not in t.shape for t in self.input_tensors])
        ]

    @property
    def I(self) -> List[str]:
        syms = []
        for t in self.input_tensors:
            syms.extend(t.shape)
        return syms

    @property
    def O(self) -> List[str]:
        return list(self.output_symb_2_value.keys())

    def _concretize_input_arg(
        self,
        value: AbsValue,
        symb_2_value: Dict[str, Any] = None,
        tensor_from_numpy: Callable = lambda x: x,
    ) -> Any:
        if symb_2_value is None:
            symb_2_value = self.input_symb_2_value
        if isinstance(value, list):
            return [
                self._concretize_input_arg(v, symb_2_value, tensor_from_numpy)
                for v in value
            ]
        else:
            return value.concretize(symb_2_value, tensor_from_numpy)

    def input_args(
        self,
        symb_2_value: Dict[str, Any] = None,
        tensor_from_numpy: Callable = lambda x: x,
    ):
        if symb_2_value is None:
            symb_2_value = self.input_symb_2_value
        args, kwargs = [], {}
        for name, is_pos, abs_value in zip(self.names, self.is_pos, self.abs_values):
            value = self._concretize_input_arg(
                abs_value, symb_2_value, tensor_from_numpy
            )
            if is_pos:
                args.append(value)
            else:
                kwargs[name] = value
        return args, kwargs

    def concrete_input_tensors(
        self,
        symb_2_value: Dict[str, Any] = None,
        tensor_from_numpy: Callable = lambda x: x,
    ):
        if symb_2_value is None:
            symb_2_value = self.input_symb_2_value
        return [
            t.concretize(symb_2_value, tensor_from_numpy) for t in self.input_tensors
        ]

    def materialize(self, func: Callable, attr_map: Dict[str, int]) -> Callable:
        """
        Usage:
            tensor_op = inst.materialize(eval(inst.name), attr_map)
            ret = tensor_op(tensor1, tensot2,)
        """

        tplaces = []

        def concretize_arg(value: AbsValue, k, indices) -> Any:
            if isinstance(value, list):
                return [
                    concretize_arg(v, k, indices + [i]) for i, v in enumerate(value)
                ]
            elif isinstance(value, AbsTensor):
                tplaces.append((k, indices))
                return value
            else:
                return value.concretize(attr_map)

        args, kwargs = [], {}
        for name, is_pos, abs_value in zip(self.names, self.is_pos, self.abs_values):
            k = name if not is_pos else len(args)
            value = concretize_arg(abs_value, k, [])
            if is_pos:
                args.append(value)
            else:
                kwargs[name] = value

        def invoke(args, kwargs, tplaces, *tensors):
            tensors = list(tensors)
            for (k, indices), tensor in zip(tplaces, tensors):
                if isinstance(k, int):
                    if len(indices) == 0:
                        args[k] = tensor
                    else:
                        reduce(list.__getitem__, indices[:-1], args[k])[
                            indices[-1]
                        ] = tensor
                else:
                    if len(indices) == 0:
                        kwargs[k] = tensor
                    else:
                        reduce(list.__getitem__, indices[:-1], kwargs[k])[
                            indices[-1]
                        ] = tensor
            return func(*args, **kwargs)

        return partial(invoke, args, kwargs, tplaces)

    def invoke_str(self, attr_map):
        args, kwargs = [], {}

        def strr(x):
            if isinstance(x, str) and x != "??":
                return f"'{x}'"
            return str(x)

        def concretize_input_arg(value: AbsValue) -> Any:
            if isinstance(value, list):
                return [concretize_input_arg(v) for v in value]
            elif isinstance(value, AbsTensor):
                return "??"
            else:
                return value.concretize(attr_map)

        for name, is_pos, abs_value in zip(self.names, self.is_pos, self.abs_values):
            value = concretize_input_arg(abs_value)
            if is_pos:
                args.append(value)
            else:
                kwargs[name] = value

            arg_str = ", ".join(map(strr, args))
            kwargs_str = ", ".join(f"{k}={strr(v)}" for k, v in kwargs.items())

        if arg_str and kwargs_str:
            return f"{self.name}({arg_str}, {kwargs_str})"
        elif arg_str:
            return f"{self.name}({arg_str})"
        else:
            return f"{self.name}({kwargs_str})"

    def concrete_input_shapes(self, symb_2_value: Dict[str, Any] = None):
        if symb_2_value is None:
            symb_2_value = self.input_symb_2_value
        return [
            abs_tensor.concrete_shape(symb_2_value) for abs_tensor in self.input_tensors
        ]

    def concrete_output_shapes(self, symb_2_value: Dict[str, Any] = None):
        if symb_2_value is None:
            symb_2_value = self.output_symb_2_value
        return [
            abs_tensor.concrete_shape(symb_2_value)
            for abs_tensor in self.output_tensors
        ]

    @property
    def input_tensor_dtypes(self) -> List[str]:
        return [t.dtype for t in self.input_tensors]

    @property
    def output_tensor_dtypes(self) -> List[str]:
        return [t.dtype for t in self.output_tensors]

    @property
    def abs_record(self) -> Tuple:
        return (
            self.input_symb_2_value,
            self.output_symb_2_value,
            self.input_tensor_dtypes,
            self.output_tensor_dtypes,
        )

    def _parse_output_value(
        self, value, symb_2_value: Dict[str, Any], output_tensors: List[AbsTensor]
    ) -> Any:
        if isinstance(value, list):
            return [
                self._parse_output_value(v, symb_2_value, output_tensors) for v in value
            ]
        elif isinstance(value, np.ndarray):
            abs_tensor = AbsTensor.from_numpy(value)
            for i_s, s in enumerate(abs_tensor.shape):
                symb = f"o{len(symb_2_value)}"
                symb_2_value[symb] = s
                abs_tensor.shape[i_s] = symb
            output_tensors.append(abs_tensor)
            return abs_tensor
        else:
            return value

    def output_info(self, out_list) -> Tuple[Dict[str, Any], List[AbsTensor]]:
        symb_2_value: Dict[str, Any] = {}
        output_tensors: List[AbsTensor] = []
        output_values = self._parse_output_value(out_list, symb_2_value, output_tensors)
        return symb_2_value, output_tensors

    def make_tensors_numeric(self) -> None:
        for ts in self.input_tensors + self.output_tensors:
            if ts.dtype == "object":
                ts.dtype = "float32"

    def rule_init(self):
        if hasattr(self, "rule_init_flag"):
            return
        self.rule_init_flag = True
        self.type_transfer_rules, self.type_transfer_dbg_info = gen_type_transfer_trees(
            self
        )
        self.requires_rules, self.requires_dbg_info = gen_requires_trees(self)
        self.nnsmith_rules_list = gen_nnsmith_rules(self)
        self.infer_fail = judge_failure(self)

    def infer_failed(self):
        self.rule_init()
        return self.infer_fail

    def type_transfer_dbg(self):
        self.rule_init()
        return self.type_transfer_dbg_info

    def requires_dbg(self):
        self.rule_init()
        return self.requires_dbg_info

    def type_transfer_expressions(self, inputs):
        self.rule_init()
        outputs = defaultdict(list)
        for symb, rules in self.type_transfer_rules.items():
            for rule_tree in rules:
                outputs[symb].append(rule_tree.nnsmith_evaluate(inputs))
        return outputs

    def requires_expressions(self, inputs):
        self.rule_init()
        outputs = []
        for rule_tree, sign in self.requires_rules:
            expr = rule_tree.nnsmith_evaluate(inputs)
            if sign == "==":
                outputs.append(expr == 0)
            elif sign == ">=":
                outputs.append(expr >= 0)
            elif sign == ">":
                outputs.append(expr > 0)
        return outputs

    def nnsmith_rules(self):
        self.rule_init()
        from dynofuzz.autoinf import ATTR_FREE_RULES

        return [ATTR_FREE_RULES[i] for i in self.nnsmith_rules_list]

    # def execute(
    #     self,
    #     symb_2_value: Dict[str, Any] = None,
    #     tensor_from_numpy: Callable = lambda x: x,
    #     numpy_from_tensor: Callable = lambda x: x,
    #     is_tensor: Callable = lambda x: False,
    # ) -> Tuple[Dict[str, Any], List[AbsTensor]]:
    #     if symb_2_value is None:
    #         symb_2_value = self.input_symb_2_value
    #     func = eval(self.name)
    #     args, kwargs = self.input_args(symb_2_value, tensor_from_numpy)
    #     ret = func(*args, **kwargs)
    #     ret_list = get_ret_list(ret)
    #     ret_list = [numpy_from_tensor(r) if is_tensor(r) else r for r in ret_list]
    #     return self.output_info(ret_list)
