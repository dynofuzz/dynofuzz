import logging
import random
import time
import traceback
from abc import abstractmethod
from itertools import product
from typing import Dict, List, Optional, Set, Tuple, Type

import z3

from dynofuzz.abstract.arith import *
from dynofuzz.abstract.dtype import *
from dynofuzz.abstract.op import (
    AbsOpBase,
    AbsTensor,
    Expand,
    Placeholder,
    concretize_op,
    rank_all,
)
from dynofuzz.autoinf import AutoInfOpBase, OpInstance, OpRecordFinder
from dynofuzz.error import ConstraintCheck, ConstraintError, SanityCheck
from dynofuzz.gir import GraphIR, InstExpr, InstIR
from dynofuzz.logging import MGEN_LOG, SMT_LOG
from dynofuzz.util import HAS_PYGRAPHVIZ, set_seed, viz_dot


class RequiredDimNotFound(Exception):
    pass


def concretize_op_autoinf(op, model):
    if isinstance(op, AutoInfOpBase):
        attrs = {}
        for k, v in op.attrs.items():
            e_val = model.eval(v)
            if isinstance(e_val, z3.IntNumRef):
                attrs[k] = e_val.as_long() if isinstance(v, z3.ExprRef) else v
            else:
                attrs[k] = random.randint(0, 64)  # not inferred -> no constraints.
        op = AutoInfOpBase(inst=op.inst, attrs=attrs)
    else:
        op = concretize_op(op, model)
    return op


def concretize_graph(ir: GraphIR, model: z3.ModelRef) -> GraphIR:
    return ir.concretize(model)


class BaseGen:
    def __init__(
        self,
        opset,
        seed=None,
        forward_prob=None,
        concr_ph_dim_rng=(1, 64),
        max_elem_per_tensor=2**16,
    ):
        assert len(opset) > 0, "opset must not be empty"
        if seed is not None:
            set_seed(seed)

        self.seed = seed
        self.op_candidates = opset
        self.ir = GraphIR()
        self.monotonic_placeholder_id = 0

        # Names of current placeholders
        self.placeholders: List[str] = []
        # for all (including newly created tmp) placeholders
        self.forward_prob = 0.5 if forward_prob is None else forward_prob
        self.concr_ph_dim_rng = concr_ph_dim_rng
        self.max_elem_per_tensor = max_elem_per_tensor
        self.acc_smt_time_ms = 0

    def random_rank(self):
        return random.choice(rank_all())

    def tensor_type_constraints(
        self, atensor: AbsTensor
    ) -> List[Union[z3.BoolRef, bool]]:
        return [atensor.nelement() <= self.max_elem_per_tensor]

    @abstractmethod
    def assume(self, c: Union[z3.BoolRef, bool]):
        pass

    def make_symbolic_placeholder(self, rank, dtype=None) -> Placeholder:
        syms = self.new_syms(
            [f"ph{self.monotonic_placeholder_id}_{k}" for k in range(rank)]
        )
        ph = Placeholder(
            AbsTensor(
                shape=syms, dtype=dtype if dtype is not None else self.random_dtype()
            )
        )
        self.monotonic_placeholder_id += 1
        return ph

    def make_random_concrete_placeholder(self, rank, dtype=None):
        l, r = self.concr_ph_dim_rng
        shape = []
        product = 1
        for _ in range(rank):
            v = random.randint(l, r)
            if product * v > self.max_elem_per_tensor:
                v = 1
            shape.append(v)
            product *= v

        # shuffle
        random.shuffle(shape)

        ph = Placeholder(
            AbsTensor(
                shape=shape,
                dtype=dtype if dtype is not None else self.random_dtype(),
            )
        )
        return ph

    def random_dtype(self):
        # more floats than ints.
        wts = [1] * len(DTYPE_GEN_ALL)
        for i in DTYPE_GEN_FLOATS:
            wts[DTYPE_GEN_ALL.index(i)] = 4
        for i in DTYPE_GEN_INTS:
            wts[DTYPE_GEN_ALL.index(i)] = 1
        return random.choices(DTYPE_GEN_ALL, weights=wts)[0]

    def new_sym(self, name):
        return z3.Int(name)

    def new_syms(self, names):
        return [self.new_sym(name) for name in names]

    def insert_init_ph_node(self, ph: Placeholder) -> InstIR:
        inst = self.forward_insert_node(ph, [])

        for c in ph.ttype.sym_gt_conc_ge_zero():
            self.assume(c)

        return inst

    @abstractmethod
    def try_forward_insert_at(self, node: AbsOpBase, input_vars: List[str]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def try_occupy_placeholder(self, node: AbsOpBase, phvars: List[str]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def make_concrete(self) -> GraphIR:
        raise NotImplementedError

    def extra_exit_check(self, max_node_size) -> bool:
        """
        Returns:
            bool: add more checks to determine whether to exit the generation.
        """
        return False

    def num_op(self) -> int:
        # exclude placeholders.
        return self.ir.n_compute_inst()

    def try_insert(self):
        node_t = self.pick_next_op_type()
        self.try_insert_node_type(node_t)

    def abstract_gen(self, max_node_size=10, max_gen_millisec=2000):
        z3.set_param("timeout", max_gen_millisec // 3)

        assert max_node_size > 0, "max_node_size must be positive"

        init_time = time.time()

        # starts generation.
        while (
            time.time() - init_time < max_gen_millisec / 1000
            and self.num_op() < max_node_size
        ):
            if self.extra_exit_check(max_node_size):
                break
            self.try_insert()

        # init graph placeholders
        SanityCheck.gt(len(self.placeholders), 0)

        def determine_ph_type(ph: str, to_input: bool):
            SanityCheck.true(ph in self.placeholders)
            ph_inst_id, _ = InstIR.var_inst_idx(ph)
            ph_inst = self.ir.find_inst_by_id(ph_inst_id)
            if to_input:
                ph_inst.iexpr.op = ph_inst.iexpr.op.to_input()
            else:
                ph_inst.iexpr.op = ph_inst.iexpr.op.to_const()

        determine_ph_type(self.placeholders[0], True)  # At lease make one input.
        for ph in self.placeholders[1:]:
            determine_ph_type(ph, random.randint(0, 1))

    def pick_next_op_type(self):
        return random.choice(self.op_candidates)

    def forward_insert_node(self, node: AbsOpBase, input_vars: List[str]) -> InstIR:
        new_inst = self.ir.add_inst(InstExpr(op=node, args=tuple(input_vars)))

        if isinstance(node, Placeholder):
            # Add placeholder's return varname.
            self.placeholders.append(new_inst.retval())

        return new_inst

    def backward_insert_node(
        self, node, input_vars: List[str], ph_to_replace: List[str]
    ) -> InstIR:
        new_inst = self.forward_insert_node(node, input_vars=input_vars)

        # replace all uses of ph_to_replace
        # and delete the unused placeholders.
        for ph, rv in zip(ph_to_replace, new_inst.retvals()):
            self.ir.replace_alluse(ph, rv)
            ph_inst_id, _ = InstIR.var_inst_idx(ph)
            ph_inst = self.ir.find_inst_by_id(ph_inst_id)
            self.ir.remove_unused(ph_inst)
            self.placeholders.remove(ph)

        return new_inst

    def try_forward_insert(self, op: AbsOpBase) -> bool:
        n_inp = len(op.inp_ranks)
        dim_spec_list = []

        if op.same_inp_dims:  # find `n_inp` under the same input shapes.
            rank_set = set(op.inp_ranks[0])

            for ranks in op.inp_ranks[1:]:
                rank_set.intersection_update(set(ranks))

            SanityCheck.ge(len(rank_set), 1)

            final_dim = random.choice(list(rank_set))
            dim_spec_list = [(final_dim,)] * n_inp
        else:  # inputs have different dimension sizes.
            dim_spec_list = op.inp_ranks

        varnames = self.pick_var_group(
            dim_spec_list,
            op.in_dtypes,
        )

        if self.try_forward_insert_at(op, varnames):
            return True

        return False

    def try_backward_insert(self, op: AbsOpBase):
        # we know that: Y = op(X)
        # S1 - select Y: Y must be a placeholder; (this also means the graph must start w/ a placeholder)
        phvars = self.pick_var_group(
            op.out_ranks,
            op.out_dtypes,
            var_candidates=[
                name
                for name in self.placeholders
                if not isinstance(op, Expand)
                or self.ir.vars[name].ndims < op.expand_last_dim
            ],
        )

        if self.try_occupy_placeholder(op, phvars):
            return True

        return False

    def try_insert_node_type(
        self, node_t: Type[AbsOpBase], max_tensor_pick_time=3
    ) -> bool:
        MGEN_LOG.debug(
            f"@[Node #{self.ir.n_inst()}] <-- trying to insert node type {node_t.__name__}"
        )

        try:
            for _ in range(max_tensor_pick_time):
                # should recreate a new instance since some attributes (like axis) should be initialized for each pick
                op_param_n = node_t.get_num_var_param()
                op_id = self.ir.n_inst()
                op_params = [
                    self.new_sym("op%s_%s" % (op_id, k)) for k in range(op_param_n)
                ]

                op: AbsOpBase = node_t(*op_params)

                if random.uniform(0, 1) < self.forward_prob:
                    if self.try_forward_insert(op):
                        return True
                else:
                    if self.try_backward_insert(op):
                        return True
        except RequiredDimNotFound:
            if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
                MGEN_LOG.debug(traceback.format_exc())
            return False
        except ConstraintError:
            if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
                MGEN_LOG.debug(traceback.format_exc())
            return False

        return False

    def filter_rank_dtype(self, ndims, dtype, candidates: List[str]):
        cans = candidates

        cans = list(
            filter(  # filter with ndim
                lambda vname: self.ir.vars[vname].ndims in ndims, cans
            )
        )
        if len(cans) == 0:
            raise RequiredDimNotFound(f"Cannot find candidate to sat rank of {ndims}.")

        if dtype is not None:
            cans = list(
                filter(  # filter with dtype
                    lambda vname: self.ir.vars[vname].dtype == dtype, cans
                )
            )
            if len(cans) == 0:
                raise RequiredDimNotFound(
                    f"Cannot find candidate to sat rank of {ndims} and dtype {dtype}."
                )

        return cans

    def pick_var_group(
        self,
        ndim_list: List[Set[int]],
        dtype_combs_spec: List[Tuple[DType, ...]],
        var_candidates: Optional[List[str]] = None,
    ) -> List[str]:
        """Randomly pick a group of variables that satisfy one of the `dtype_combs_spec` and `ndim_list`.

        Returns:
            List[str]: Satisfiable group of variable names.
        """

        if var_candidates is None:
            var_candidates = list(self.ir.vars.keys())

        # check if can gen var group data types:
        dtypes_in_ir = set([self.ir.vars[vname].dtype for vname in var_candidates])
        if dtypes_in_ir.isdisjoint(set(DTYPE_GEN_ALL)):
            raise RequiredDimNotFound(
                f"DType unsat in IR (possibly due to complex64/128 dtypes)."
            )

        abs_tensor_candidates = []
        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            for cand in var_candidates:
                MGEN_LOG.debug(
                    f"Candidate: {cand}: {self.ir.vars[cand].dtype} ~ {self.ir.vars[cand].ndims}"
                )
            MGEN_LOG.debug(f"Input data ranks candidates: {ndim_list}")
            MGEN_LOG.debug(f"Input data types candidates: {dtype_combs_spec}")

        viable_dtypes = []
        for i, ndims in enumerate(ndim_list):
            viable_dtypes.extend(
                [
                    self.ir.vars[vname].dtype
                    for vname in self.filter_rank_dtype(
                        ndims=ndims, dtype=None, candidates=var_candidates
                    )
                ]
            )
        # only use dtypes currently available after ndim filtering
        dtype_combs = [
            comb for comb in dtype_combs_spec if all(dt in viable_dtypes for dt in comb)
        ]
        if len(dtype_combs) == 0:
            raise RequiredDimNotFound(
                f"No viable candidates: rank within {ndim_list} and dtype within {dtype_combs_spec}."
            )
        dtype_comb = random.choice(dtype_combs)
        for i, ndims in enumerate(ndim_list):
            candidates = self.filter_rank_dtype(
                ndims=ndims, dtype=dtype_comb[i], candidates=var_candidates
            )
            abs_tensor_candidates.append(random.choice(candidates))

        return abs_tensor_candidates

    def base_check_sat(self, solver: z3.Solver, *assumptions) -> z3.CheckSatResult:
        start = time.time()

        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            if solver.assertions():
                SMT_LOG.debug(
                    f"existing constraints: {', '.join(map(str, solver.assertions()))}"
                )
            if assumptions:
                SMT_LOG.debug(f"new constraints: {', '.join(map(str, assumptions))}")

        cres = solver.check(*assumptions)

        smt_time = int((time.time() - start) * 1000)
        self.acc_smt_time_ms += smt_time

        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            SMT_LOG.debug(f"{cres} <-- checking time: {smt_time} ms")

            if cres == z3.unsat:
                SMT_LOG.debug(f"Unsat core: {solver.unsat_core()}")

        return cres


def set_z3_state(seed=None):
    z3.set_param(
        "smt.phase_selection",
        5,
        "smt.arith.random_initial_value",
        True,
        "smt.random_seed",
        seed,
        "sat.random_seed",
        seed,
        "sat.phase",
        "random",
        "memory_max_size",
        50 * 1024,  # MB
    )


class SymbolicGen(BaseGen):
    def __init__(
        self,
        opset,
        seed=None,
        init_fp=False,
        symbolic_init=True,
        **kwargs,
    ):
        super().__init__(opset, seed, **kwargs)
        if seed is not None:
            set_z3_state(seed)

        self.solver = z3.Solver()
        self.last_solution: Optional[z3.ModelRef] = None

        # Insert the first node.
        if symbolic_init:
            ph = self.make_symbolic_placeholder(
                self.random_rank(), dtype=DType.float32 if init_fp else None
            )
        else:
            ph = self.make_random_concrete_placeholder(
                self.random_rank(), dtype=DType.float32 if init_fp else None
            )

        self.insert_init_ph_node(ph)
        for pred in self.tensor_type_constraints(ph.ttype):
            self.assume(pred)

    def assume(self, c: z3.BoolRef):
        self.solver.add(c)

    def check_sat(self, *assumptions):
        cres = self.base_check_sat(self.solver, *assumptions)
        if cres == z3.sat:
            self.last_solution = self.solver.model()
        return cres

    def try_forward_insert_at(self, node: AbsOpBase, input_vars: List[str]) -> bool:
        itensors = [self.ir.vars[vname] for vname in input_vars]
        constraints = node.checked_requires(itensors)

        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            SMT_LOG.debug(f"---> Trying to solve: {node} ~ {constraints}")

        # make a copy
        otensors = node.checked_type_transfer(itensors)

        for aten in otensors:
            for c in aten.gt_zero():
                constraints.append(c)

        # limit output tensor size
        for aten in otensors:
            constraints.extend(self.tensor_type_constraints(aten))

        check_res = self.check_sat(*constraints)

        if check_res != z3.sat:
            return False

        for c in constraints:
            self.assume(c)

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Forward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {itensors}")
            MGEN_LOG.debug(f"\toutputs: {otensors}")

        node.bind_input_like(itensors)
        node.bind_output_like(otensors)

        self.forward_insert_node(node, input_vars)
        return True

    def try_occupy_placeholder(self, node: AbsOpBase, phvars: List[str]) -> bool:
        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(
                f"---> Trying to occupy placeholder: {phvars} for node {node}"
            )
        # S2 - create X: X can be
        #                   - a new placeholder (fallback)
        #                   - an existing alive shape

        otensors = [self.ir.vars[name] for name in phvars]

        # S2.2: try to reuse some existing outputs;
        # TODO: allow reuse existing alive shapes
        # n_inps = len(node.inp_ranks)
        # max_try = 2
        # n_reuse = n_inps - 1
        # while n_reuse > 0 and max_try > 0:
        #     # TODO...
        #     max_try -= 1
        #     n_reuse -= 1

        # S2.2: reusing outputs failed. as a fallback, promote all free vars to placeholders.
        phs_as_op_inputs: List[Placeholder] = []
        constraints = []
        for rank, dtype in node.deduct_inp_ranks_and_dtype(otensors):
            # oversample rank 4 tensors as they may be more important
            ph = self.make_symbolic_placeholder(
                rank if rank != -1 else self.random_rank(), dtype=dtype
            )
            phs_as_op_inputs.append(ph)
            constraints.extend(ph.ttype.gt_zero())
            constraints.extend(self.tensor_type_constraints(ph.ttype))

        itensors = [p.ttype for p in phs_as_op_inputs]
        constraints.extend(node.checked_requires(itensors))
        inferred_otensors = node.checked_type_transfer(itensors)

        for i, shape in enumerate(inferred_otensors):
            constraints.extend(shape.eq(otensors[i]))
            constraints.extend(shape.gt_zero())

        check_res = self.check_sat(*constraints)

        if check_res != z3.sat:
            return False

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Backward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {phs_as_op_inputs}")

        for c in constraints:
            self.assume(c)

        # succ.
        input_vars = []

        for ph in phs_as_op_inputs:
            inst = self.forward_insert_node(ph, [])
            input_vars.append(inst.retval())

        node.bind_input_like(itensors)
        node.bind_output_like(inferred_otensors)

        self.backward_insert_node(node, input_vars, phvars)

        return True

    def make_concrete(self) -> GraphIR:
        SanityCheck.not_none(self.last_solution, "Run check_sat first!")
        self.ir.concretize(self.last_solution)
        return self.ir


class ConcolicGen(BaseGen):
    """Different from SymbolicGen, the graph after an insertion is `concrete` in ConcolicGen.
    However, each step when inserting a node, we symbolically find a satisfiable solution for it."""

    def __init__(
        self,
        opset,
        seed=None,
        init_fp=False,
        **kwargs,
    ):
        super().__init__(opset, seed, **kwargs)
        if seed is not None:
            set_z3_state(seed)

        # Insert the first node.
        self.insert_init_ph_node(
            self.make_random_concrete_placeholder(
                self.random_rank(), dtype=DType.float32 if init_fp else None
            )
        )

    def try_forward_insert_at(
        self, node: AbsOpBase, input_vars: List[str], force_positive_dim=False
    ) -> bool:
        solver = z3.Solver()

        itensors = [self.ir.vars[vname] for vname in input_vars]
        constraints = node.checked_requires(itensors)

        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            SMT_LOG.debug(f"---> Trying to solve: {node} ~ {constraints}")

        if force_positive_dim:
            for aten in itensors:
                if aten.gt_zero():
                    return False

        # make a copy
        otensors = node.checked_type_transfer(itensors)

        if force_positive_dim:
            for aten in otensors:
                for c in aten.gt_zero():
                    constraints.append(c)
        else:
            for aten in otensors:
                for c in aten.sym_gt_conc_ge_zero():
                    constraints.append(c)

        check_res = self.base_check_sat(solver, *constraints)

        if check_res != z3.sat:
            return False

        # materialize otensors and attributes.
        node = concretize_op_autoinf(node, solver.model())
        otensors = node.checked_type_transfer(itensors)

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Forward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {itensors}")
            MGEN_LOG.debug(f"\toutputs: {otensors}")

        # Shape checker.
        # NOTE: No need to check input shape as they are already checked for being in the graph.
        for i, ten in enumerate(otensors):
            if not all(self.tensor_type_constraints(ten)):
                MGEN_LOG.debug(f"{i}-th output type constraint failed: {ten}")
                return False

        node.bind_input_like(itensors)
        node.bind_output_like(otensors)

        self.forward_insert_node(node, input_vars)
        return True

    def try_occupy_placeholder(
        self, node: AbsOpBase, phvars: List[str], force_positive_dim=False
    ) -> bool:
        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(
                f"---> Trying to occupy placeholder: {phvars} for node {node}"
            )

        # TODO: In backward insertion, reusing existing tensors is not implemented.

        # Concrete tensors.
        solver = z3.Solver()

        otensors = [self.ir.vars[name] for name in phvars]
        phs_as_op_inputs: List[Placeholder] = []
        constraints = []
        for rank, dtype in node.deduct_inp_ranks_and_dtype(otensors):
            # oversample rank 4 tensors as they may be more important
            ph = self.make_symbolic_placeholder(
                rank if rank != -1 else self.random_rank(), dtype=dtype
            )
            phs_as_op_inputs.append(ph)
            constraints.extend(
                ph.ttype.gt_zero()
                if force_positive_dim
                else ph.ttype.sym_gt_conc_ge_zero()
            )

        itensors = [p.ttype for p in phs_as_op_inputs]
        constraints.extend(node.checked_requires(itensors))
        inferred_otensors = node.checked_type_transfer(itensors)

        for i, shape in enumerate(inferred_otensors):
            constraints.extend(shape.eq(otensors[i]))
            constraints.extend(
                shape.gt_zero() if force_positive_dim else shape.sym_gt_conc_ge_zero()
            )

        check_res = self.base_check_sat(solver, *constraints)

        if check_res != z3.sat:
            return False

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug(f">> Backward insert: {node}")
            MGEN_LOG.debug(f"\tinputs:  {phs_as_op_inputs}")

        model = solver.model()
        # succ.
        itensors = []
        for i, ph in enumerate(phs_as_op_inputs):
            # materialize ph
            phs_as_op_inputs[i] = concretize_op_autoinf(ph, model)
            itensors.append(phs_as_op_inputs[i].ttype)

        # Input Shape checker.
        # NOTE: No need to check output because they are already in the graph thus valid.
        for i, ten in enumerate(itensors):
            if not all(self.tensor_type_constraints(ten)):
                MGEN_LOG.debug(f"{i}-th input type constraint failed: {ten}")
                return False

        node = concretize_op_autoinf(node, model)
        node.bind_input_like(itensors)
        node.bind_output_like(otensors)

        # Apply insertion.
        input_vars = []
        for ph in phs_as_op_inputs:
            inst = self.forward_insert_node(ph, [])
            input_vars.append(inst.retval())

        self.backward_insert_node(node, input_vars, phvars)

        return True

    def assume(self, c: bool):
        # semantically equivalent to `assert c`.
        ConstraintCheck.true(c, "Assumption failed")

    def make_concrete(self) -> GraphIR:
        return self.ir


MAX_PRODUCT_COMPLEXITY = 2**16


class DynoFuzzR(ConcolicGen):
    """Concolic generation assisted with concrete records."""

    def __init__(
        self,
        opset,
        record_finder: OpRecordFinder,
        seed=None,
        init_fp=False,
        **kwargs,
    ):
        BaseGen.__init__(self, opset, seed, **kwargs)
        if seed is not None:
            set_z3_state(seed)

        # remove records whose tensors violate tensor_type_constraints
        # FIXME: Strictly apply tensor_type_constraints to filter unapplicable tensors.
        self.record_finder = record_finder

        # Insert the first node.
        self.forward_insert_node(
            self.make_random_concrete_placeholder(
                self.random_rank(), dtype=DType.float32 if init_fp else None
            ),
            [],
        )

    def make_random_concrete_placeholder(self, rank, dtype=None):
        cand: List[AbsTensor] = []
        for tensor in self.record_finder.producer.keys():
            if tensor.ndims == rank and (dtype is None or tensor.dtype == dtype):
                cand.append(tensor)
        for tensor in self.record_finder.consumer.keys():
            if tensor.ndims == rank and (dtype is None or tensor.dtype == dtype):
                cand.append(tensor)

        if len(cand) == 0:
            MGEN_LOG.warning(f"No record w/ rank@{rank}<{dtype}>. Fallback to base.")
            return super().make_random_concrete_placeholder(rank, dtype)

        selected = random.choice(cand)

        ph = Placeholder(
            AbsTensor(
                shape=selected.shape,
                dtype=selected.dtype,
            )
        )
        return ph

    def try_concrete_forward_insert_op(
        self, type2vars, op: AbsOpBase, itensors, otensors
    ) -> bool:
        ivars = []
        for it in itensors:
            if it not in type2vars:
                break
            ivars.append(random.choice(type2vars[it]))
        if len(ivars) == len(itensors):
            # forward insert
            op.bind_input_like(itensors)
            op.bind_output_like(otensors)
            self.forward_insert_node(op, ivars)
            return True
        return False

    def try_concrete_backward_insert_op(
        self, type2vars: Dict[AbsTensor, List[str]], op: AbsOpBase, itensors, otensors
    ) -> bool:
        type2phvars = {
            k: [v for v in vs if v in self.placeholders] for k, vs in type2vars.items()
        }
        # remove k with empty list
        type2phvars = {k: vs for k, vs in type2phvars.items() if len(vs) > 0}

        ovars = []
        for ot in otensors:
            if ot not in type2phvars:
                break
            # Cannot use the same variable twice.
            cands = [v for v in type2phvars[ot] if v not in ovars]
            if len(cands) == 0:
                break
            ovars.append(random.choice(cands))

        if len(ovars) == len(otensors):
            # backward insert
            # create placeholder.
            op_ivars = []

            for ttype in itensors:
                inst = self.forward_insert_node(Placeholder(ttype=ttype), [])
                op_ivars.append(inst.retval())

            op.bind_input_like(itensors)
            op.bind_output_like(otensors)
            self.backward_insert_node(op, op_ivars, ovars)
            return True

        return False

    def try_concrete_insert_forward(self):
        # Analyze the graph:
        # 1. self.vars: ~ all tensor variables.
        # 2. [Coarse-grained search] find records which has such variables types.
        op_types = []

        type2vars = {}  # available types.
        for k, v in self.ir.vars.items():
            type2vars.setdefault(v, []).append(k)

        for v in type2vars:
            for ot in self.record_finder.consumer.get(v, []):
                if ot not in op_types:
                    op_types.append(ot)  # operators that needs those types.

        # 3. [Fine-grained search] find records whose inputs/outputs can be exactly matched.
        random.shuffle(op_types)

        for op_type in op_types:
            # check match.
            for itensors, otensors, attrs in self.record_finder.op2record[op_type]:
                op: AbsOpBase = AutoInfOpBase(op_type, attrs=attrs)

                try:
                    if self.try_concrete_forward_insert_op(
                        type2vars, op, itensors, otensors
                    ):
                        return True
                except ConstraintError:
                    pass

        return False

    def try_concrete_insert_backward(self):
        # Analyze the graph:
        # 1. self.vars: ~ all tensor variables.
        # 2. [Coarse-grained search] find records which has such variables types.
        op_types = []

        type2vars = {}  # available types of placeholders!
        for k in self.placeholders:
            v = self.ir.vars[k]
            type2vars.setdefault(v, []).append(k)

        for v in type2vars:
            for ot in self.record_finder.producer.get(v, []):
                if ot not in op_types:
                    op_types.append(ot)  # operators that produces those types.

        # 3. [Fine-grained search] find records whose inputs/outputs can be exactly matched.
        random.shuffle(op_types)

        for op_type in op_types:
            # check match.
            for itensors, otensors, attrs in self.record_finder.op2record[op_type]:
                op: AbsOpBase = AutoInfOpBase(op_type, attrs=attrs)

                try:
                    if self.try_concrete_backward_insert_op(
                        type2vars, op, itensors, otensors
                    ):
                        return True
                except ConstraintError:
                    pass

        return False

    def try_insert(self):
        if random.random() < 0.5:
            if random.random() < self.forward_prob:
                if self.try_concrete_insert_forward():
                    self.symbolic_impossible = 0
                    return True
            else:
                if self.try_concrete_insert_backward():
                    self.symbolic_impossible = 0
                    return True

        # This can be impossible if there is a dtype mismatch.
        dtypes_in_ir = set([t.dtype for t in self.ir.vars.values()])
        if dtypes_in_ir.isdisjoint(set(DTYPE_GEN_ALL)):
            self.symbolic_impossible += 1
            return False

        return BaseGen.try_insert(self)

    def extra_exit_check(self, max_node_size):
        # Check if all tensors are used.
        return self.symbolic_impossible >= max_node_size

    def abstract_gen(self, max_node_size=10, max_gen_millisec=2000):
        self.symbolic_impossible = 0
        return BaseGen.abstract_gen(self, max_node_size, max_gen_millisec)


class DynoFuzz(DynoFuzzR):
    def try_insert(self):
        meta_selector = random.random()  # [0, 1]
        if meta_selector < 0.33:
            if random.random() < self.forward_prob:
                if self.try_concrete_insert_forward():
                    self.symbolic_impossible = 0
                    return True
            else:
                if self.try_concrete_insert_backward():
                    self.symbolic_impossible = 0
                    return True
        elif meta_selector < 0.66:
            if random.random() < self.forward_prob:
                if self.try_autoinf_insert_forward():
                    self.symbolic_impossible = 0
                    return True
            else:
                if self.try_autoinf_insert_backward():
                    self.symbolic_impossible = 0
                    return True

        # This can be impossible if there is a dtype mismatch.
        dtypes_in_ir = set([t.dtype for t in self.ir.vars.values()])
        if dtypes_in_ir.isdisjoint(set(DTYPE_GEN_ALL)):
            self.symbolic_impossible += 1
            return False

        return BaseGen.try_insert(self)

    def try_autoinf_insert_forward(self, op_bound=64) -> bool:
        htype2vars = {}  # available types.
        inst_candidates: List[OpInstance] = []

        for k, v in self.ir.vars.items():
            htype2vars.setdefault(v.htype(), []).append(k)
            for inst in self.record_finder.ihtype2op.get(v.htype(), []):
                if inst not in inst_candidates:
                    inst_candidates.append(inst)

        random.shuffle(inst_candidates)

        for inst in inst_candidates[:op_bound]:
            iranks = [t.rank for t in inst.input_tensors]
            idtype_combs = list(inst.dtype_i2o.keys())
            random.shuffle(idtype_combs)

            for idtypes in idtype_combs:
                ihtypes = [(dt, r) for dt, r in zip(idtypes, iranks)]
                if all([t in htype2vars for t in ihtypes]):
                    # Go insert.
                    op: AbsOpBase = AutoInfOpBase(
                        inst, attrs={k: self.new_sym(k) for k in inst.A}
                    )

                    # approx complexity: |t|**|ihtypes|
                    # to make |slot|**|ihtypes| < MAX_PRODUCT_COMPLEXITY
                    # => limit |slot| < MAX_PRODUCT_COMPLEXITY**(1/|ihtypes|)
                    slotmax = max(int(MAX_PRODUCT_COMPLEXITY ** (1 / len(ihtypes))), 1)
                    vcombs = list(
                        product(
                            *[
                                random.sample(
                                    htype2vars[t], min(slotmax, len(htype2vars[t]))
                                )
                                for t in ihtypes
                            ]
                        )
                    )

                    random.shuffle(vcombs)
                    for vcomb in vcombs[:4]:
                        try:
                            if self.try_forward_insert_at(
                                op, vcomb, force_positive_dim=True
                            ):
                                return True
                        except ConstraintError:
                            pass

        return False

    def try_autoinf_insert_backward(self, op_bound=64):
        htype2phs = {}  # available types.
        inst_candidates: List[OpInstance] = []

        for k in self.placeholders:
            v = self.ir.vars[k]
            htype2phs.setdefault(v.htype(), []).append(k)
            for inst in self.record_finder.ohtype2op.get(v.htype(), []):
                if inst not in inst_candidates:
                    inst_candidates.append(inst)

        random.shuffle(inst_candidates)

        for inst in inst_candidates[:op_bound]:
            oranks = [t.rank for t in inst.output_tensors]
            odtype_combs = list(inst.dtype_o2i.keys())
            random.shuffle(odtype_combs)

            for odtypes in odtype_combs:
                ohtypes = [(dt, r) for dt, r in zip(odtypes, oranks)]
                if all([t in htype2phs for t in ohtypes]):
                    op: AbsOpBase = AutoInfOpBase(
                        inst, attrs={k: self.new_sym(k) for k in inst.A}
                    )

                    slotmax = max(int(MAX_PRODUCT_COMPLEXITY ** (1 / len(ohtypes))), 1)
                    phcands = [
                        random.sample(htype2phs[t], min(slotmax, len(htype2phs[t])))
                        for t in ohtypes
                    ]

                    ph_comb_cands = [
                        c for c in product(*phcands) if len(set(c)) == len(c)
                    ]
                    random.shuffle(ph_comb_cands)

                    for phvars in ph_comb_cands[:4]:
                        try:
                            if self.try_occupy_placeholder(
                                op, phvars, force_positive_dim=True
                            ):
                                return True
                        except ConstraintError:
                            pass

        return False


class DynoFuzzI(DynoFuzz):
    def try_insert(self):
        if random.random() < 0.5:
            if random.random() < self.forward_prob:
                if self.try_autoinf_insert_forward():
                    self.symbolic_impossible = 0
                    return True
            else:
                if self.try_autoinf_insert_backward():
                    self.symbolic_impossible = 0
                    return True

        # This can be impossible if there is a dtype mismatch.
        dtypes_in_ir = set([t.dtype for t in self.ir.vars.values()])
        if dtypes_in_ir.isdisjoint(set(DTYPE_GEN_ALL)):
            self.symbolic_impossible += 1
            return False

        return BaseGen.try_insert(self)


def model_gen(
    opset: Set[Type[AbsOpBase]],
    method: str = "symbolic",
    max_nodes=5,
    seed=None,
    timeout_ms=10000,
    record_finder=None,
    **kwargs,
):
    assert max_nodes > 0, "max_nodes must >= 1"

    if "symbolic" == method or "symbolic-sinit" == method:
        gen = SymbolicGen(opset, seed, symbolic_init=True, **kwargs)
    elif "symbolic-cinit" == method:
        gen = SymbolicGen(opset, seed, symbolic_init=False, **kwargs)
    elif "concolic" == method:
        gen = ConcolicGen(opset, seed, **kwargs)
    elif "dynofuzz-r" == method:
        assert record_finder is not None, "record_finder must be provided"
        gen = DynoFuzzR(opset, record_finder, seed, **kwargs)
    elif "dynofuzz" == method:
        assert record_finder is not None, "record_finder must be provided"
        gen = DynoFuzz(opset, record_finder, seed, **kwargs)
    elif "dynofuzz-i" == method:
        assert record_finder is not None, "record_finder must be provided"
        gen = DynoFuzzI(opset, record_finder, seed, **kwargs)
    else:
        raise ValueError(f"Unknown method {method}. Try `symbolic` or `concolic`.")

    gen.abstract_gen(max_node_size=max_nodes, max_gen_millisec=timeout_ms)

    return gen


def viz(ir: GraphIR, filename: str = None):
    if HAS_PYGRAPHVIZ:
        viz_dot(ir.to_dot(), filename)
