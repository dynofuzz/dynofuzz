from math import prod
from random import randint
from typing import List, Tuple, Union

from dynofuzz.abstract.arith import *
from dynofuzz.abstract.dtype import (
    DTYPE_GEN_ALL,
    DTYPE_GEN_INTS,
    DTYPE_GEN_NON_BOOL,
    DType,
)
from dynofuzz.abstract.op import ReduceBase, UnaryOpBase, mark_materialize, rank_from
from dynofuzz.abstract.tensor import AbsTensor
from dynofuzz.error import ConstraintCheck


@mark_materialize("torch")
class Linear(UnaryOpBase):
    in_dtypes = [(DType.float32,)]
    out_dtypes = [(DType.float32,)]

    def __init__(self, ifeat: Union[int, z3.ExprRef], ofeat: Union[int, z3.ExprRef]):
        super().__init__()
        self.ifeat = ifeat
        self.ofeat = ofeat
        self.inp_ranks = [rank_from(1)]
        self.out_ranks = [rank_from(1)]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        assert len(input_shapes) == 1, "Linear only takes one input, but got {}".format(
            len(input_shapes)
        )
        return [
            AbsTensor(
                shape=[*input_shapes[0].shape[:-1], self.ofeat], dtype=DType.float32
            )
        ]

    def requires(self, input_shapes: List[AbsTensor]) -> List[z3.ExprRef]:
        ConstraintCheck.true(input_shapes[0].ndims >= 1)
        return [
            nnsmith_ge(self.ifeat, 1),
            nnsmith_ge(self.ofeat, 1),
            nnsmith_eq(input_shapes[0].shape[-1], self.ifeat),
        ]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(out_abs_tensor[0].ndims, DType.float32)]


@mark_materialize("torch")
class Flatten(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_ALL]
    out_dtypes = [(i,) for i in DTYPE_GEN_ALL]

    def __init__(self):
        super().__init__()
        self.inp_ranks = [rank_from(1)]
        self.out_ranks = [(1,)]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        inp = input_shapes[0]
        return [
            AbsTensor(
                shape=[prod(inp.shape)],
                dtype=inp.dtype,
            )
        ]

    def requires(self, input_shapes):
        return []

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(randint(0, 4), out_abs_tensor[0].dtype)]


@mark_materialize("torch")
class TorchReduceSum(ReduceBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_NON_BOOL]
    out_dtypes = [
        (i,)
        for i in DTYPE_GEN_NON_BOOL
        if i not in [DType.int8, DType.uint8, DType.int16, DType.int32]
    ]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        output = super().type_transfer(input_shapes)
        # This is a PyTorch trick...
        if input_shapes[0].dtype in DTYPE_GEN_INTS:
            output[0].dtype = DType.int64
        return output
