"""Synthesize a program from a GraphIR."""
import os
import pickle
from typing import Dict, List

import torch
import torch.nn as nn

from dynofuzz.abstract.op import Constant, Input
from dynofuzz.abstract.tensor import AbsTensor
from dynofuzz.autoinf import AutoInfOpBase
from dynofuzz.gir import GraphIR
from dynofuzz.materialize.torch.code_gen import gen_code
from dynofuzz.materialize.torch.symbolnet import random_tensor


def line(indent: int, s: str) -> str:
    return " " * indent + s


code_header = """\
from typing import Dict
import pickle
import torch
import torch.nn as nn

with open('params.pkl', 'rb') as f:
    params: Dict[str, torch.Tensor] = pickle.load(f) # 'p1': ..., 'p2': ...
with open('inputs.pkl', 'rb') as f:
    inputs: Dict[str, torch.Tensor] = pickle.load(f) # 'v1_0': ..., 'v2_0': ...
"""

code_model = """\
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.parameter.Parameter objects (with comments for shapes)
{}
        # nn.Module objects
{}

    def forward(self, {}):
{}
        return {}
"""

code_main = """\
model = Model().to({})
for k, v in inputs.items():
    inputs[k] = v.to({})
print('==== Eager mode ====')
ret_eager = model(**inputs)

print('==== {} mode ====')
{}

print('==== Check ====')
for r1, r2 in zip(ret_eager, {}):
    assert torch.allclose(r1, r2), (r1, r2)
print('OK!')
"""


class Program:
    def __init__(
        self,
        ir: GraphIR,
    ) -> None:
        self.inputs: Dict[str, torch.Tensor] = {}
        code_forward: List[str] = []
        for input_var_name in ir.input_var():
            abs_tensor: AbsTensor = ir.vars[input_var_name]
            assert abs_tensor.is_concrete(), f"Input {input_var_name} is not concrete"
            random_input = random_tensor(
                abs_tensor.shape, abs_tensor.dtype.torch(), use_cuda=False
            )  # do NOT use cuda here, or the input tensors will not be pickle.dump correctly
            self.inputs[input_var_name] = random_input
            code_forward.append(
                line(8, f"# {input_var_name}: {list(random_input.shape)}")
            )

        self.params: Dict[str, torch.Tensor] = {}
        var_2_param: Dict[str, str] = {}

        def var_names(var: List[str]) -> List[str]:
            return [var_2_param.get(v, v) for v in var]

        code_nn_modules: List[str] = []
        code_params: List[str] = []
        for ir_inst in ir.insts:
            op = ir_inst.iexpr.op
            if isinstance(op, Constant):
                p_name = f"p{len(self.params)}"
                p_data = random_tensor(
                    op.abs_tensor.shape, op.abs_tensor.dtype.torch()
                )  # do NOT use cuda here, or the tensors will not be pickle.dump correctly
                self.params[p_name] = p_data
                code_params.append(
                    line(
                        8,
                        f'self.{p_name} = torch.nn.parameter.Parameter(params["{p_name}"], requires_grad={p_data.is_floating_point()}) # {list(p_data.shape)}',
                    )
                )
                retvals = ir_inst.retvals()
                assert (
                    len(retvals) == 1
                ), f"Constant should have one retval, got {len(retvals) = }"
                var_2_param[retvals[0]] = f"self.{p_name}"
            elif not isinstance(op, Input):
                input_vals = var_names(ir_inst.iexpr.args)
                ret_vals_str = ", ".join(ir_inst.retvals())
                if isinstance(op, AutoInfOpBase):  # operator from AutoInf
                    symb_2_value = op.attrs
                    invoke_str_tmp: str = op.inst.invoke_str(symb_2_value).replace(
                        "??", "{}"
                    )
                    invoke_str = invoke_str_tmp.format(*input_vals)
                    code_forward.append(line(8, f"{ret_vals_str} = {invoke_str}"))
                else:
                    code, is_nn_module = gen_code(op)
                    if is_nn_module:
                        layer_name = f"layer{len(code_nn_modules)}"
                        code_nn_modules.append(line(8, f"self.{layer_name} = {code}"))
                        code_forward.append(
                            line(
                                8,
                                f"{ret_vals_str} = self.{layer_name}({', '.join(input_vals)})",
                            )
                        )
                    else:  # function
                        code_forward.append(
                            line(8, f"{ret_vals_str} = {code.format(*input_vals)}")
                        )
        # end for ir_inst

        bk_name = "JIT"
        code_bk_run = [
            "exported = torch.jit.trace(model, example_kwarg_inputs=inputs)",
            "exported = torch.jit.optimize_for_inference(exported)",
            "ret_exported = exported(**inputs)",
        ]
        bk_ret_name = "ret_exported"

        self.code_header = code_header
        self.code_model = code_model.format(
            "\n".join(code_params),
            "\n".join(code_nn_modules),
            ", ".join(ir.input_var()),
            "\n".join(code_forward),
            ", ".join(ir.leaf_var()),
        )
        self.code_main = code_main.format(
            'torch.device("cpu")',
            'torch.device("cpu")',
            bk_name,
            "\n".join(code_bk_run),
            bk_ret_name,
        )

    def dump(self, path: os.PathLike) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "prog.py"), "w") as f:
            print(self.code_header, file=f)
            print(self.code_model, file=f)
            print(self.code_main, file=f)
        with open(os.path.join(path, "params.pkl"), "wb") as f:
            pickle.dump(self.params, f)
        with open(os.path.join(path, "inputs.pkl"), "wb") as f:
            pickle.dump(self.inputs, f)
