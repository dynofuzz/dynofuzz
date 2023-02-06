import warnings
from typing import Dict

import numpy as np
import torch
from multipledispatch import dispatch

from dynofuzz.backends.factory import BackendCallable, BackendFactory
from dynofuzz.materialize.torch import TorchModel


class TorchCompFactory(BackendFactory):
    def __init__(self, target="cpu", optmax: bool = False):
        super().__init__(target, optmax)
        if self.target == "cpu":
            self.device = torch.device("cpu")
        elif self.target == "cuda":
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
        else:
            raise ValueError(
                f"Unknown target: {self.target}. Only `cpu` and `cuda` are supported."
            )

    @property
    def system_name(self) -> str:
        return "torchcomp"

    @dispatch(TorchModel)
    def make_backend(self, model: TorchModel) -> BackendCallable:
        torch_net = model.torch_model.to(self.device).eval()
        # trace_inp = [ts.to(self.device) for ts in torch_net.get_random_inps().values()]
        with torch.no_grad():
            with warnings.catch_warnings():
                compiled = torch.compile(
                    torch_net,
                    dynamic=False,
                    fullgraph=False,  # cannot compile SymbolNet.debug_numeric
                )

        def closure(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            input_ts = {
                k: torch.from_numpy(v).to(self.device) for k, v in inputs.items()
            }
            with torch.no_grad():
                output = compiled(*input_ts.values())
            return {
                k: v.cpu().detach().numpy()
                for k, v in zip(torch_net.output_like.keys(), output)
            }

        return closure
