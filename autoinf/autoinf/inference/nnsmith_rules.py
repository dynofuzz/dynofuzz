import multiprocessing as mp
import os
import pickle
from collections import defaultdict
from copy import deepcopy

import z3
from dynofuzz.abstract.op import *
from dynofuzz.abstract.tensor import AbsTensor
from dynofuzz.autoinf import ATTR_FREE_RULES
from dynofuzz.error import ConstraintError, InternalError

from autoinf.inference.configs import *
from autoinf.inference.invocations import input_validity_test, inst_invocations
from autoinf.inference.utils import equivalent
from autoinf.instrument.categorize import gen_inst_with_records
from autoinf.instrument.op import OpInstance
from autoinf.instrument.utils import (
    get_ret_list,
    inputs_from_record,
    tensors_from_numpys,
)


def list_to_dict(target, name):
    res = dict()
    for i, v in enumerate(target):
        res[f"{name}{i}"] = v
    return res


def nnsmith_abstensor_from_ai_abstensor(other):
    return AbsTensor(other.shape, other.dtype)


def abs_to_concrete(abs_input_tensors, concrete_values):
    concrete_input_tensors = []
    for tensor in abs_input_tensors:
        concrete_tensor = deepcopy(tensor)
        concrete_tensor.shape = list(map(concrete_values.get, tensor.shape))
        concrete_input_tensors.append(concrete_tensor)
    return concrete_input_tensors


root_dir = f"{InformationRootDir}/nnsmith_rules-{cfg_library}-{date}"


def solve_inst(opid: int, inst: OpInstance, records):
    n_input = len(inst.input_tensors)
    n_output = len(inst.output_tensors)
    success = inst_invocations(inst, "success")
    _fail = inst_invocations(inst, "fail")
    fail = []
    for inputs in _fail:
        if all(map(lambda x: x >= 0, inputs)):
            fail.append(inputs)
    # print(f'{inst.I=}')
    # print(f'{inst.A=}')
    # print(f'{inst.O=}')
    res = []
    for ind, op_type in enumerate(ATTR_FREE_RULES):
        if op_type.n_input() == n_input and op_type.n_output() == n_output:
            valid = True
            abs_input_tensors, abs_output_tensors = (
                inst.input_tensors,
                inst.output_tensors,
            )
            if op_type == Triu or op_type == Tril:
                op = op_type(diagonal=1)
            else:
                op = op_type()
            # check whether the rule can accept all successful records
            for (inputs, outputs) in success:
                input_dict, output_dict = list_to_dict(inputs, "s"), list_to_dict(
                    outputs, "o"
                )
                concrete_input_tensors = abs_to_concrete(abs_input_tensors, input_dict)
                try:
                    concrete_output_tensors = op.checked_type_transfer(
                        list(
                            map(
                                nnsmith_abstensor_from_ai_abstensor,
                                concrete_input_tensors,
                            )
                        )
                    )
                except IndexError:
                    valid = False
                except ConstraintError:
                    valid = False
                except InternalError:
                    valid = False
                else:
                    std_output_tensors = abs_to_concrete(
                        abs_output_tensors, output_dict
                    )
                    for i in range(n_output):
                        if (
                            concrete_output_tensors[i].shape
                            != std_output_tensors[i].shape
                        ):
                            valid = False
                if not valid:
                    break
                try:
                    predicates = op.checked_requires(
                        list(
                            map(
                                nnsmith_abstensor_from_ai_abstensor,
                                concrete_input_tensors,
                            )
                        )
                    )
                except IndexError:
                    valid = False
                except ConstraintError:
                    valid = False
                except InternalError:
                    valid = False
                else:
                    for predicate in predicates:
                        if not equivalent(predicate, True):
                            valid = False
                if not valid:
                    break
            if op_type == BcastBinaryOp:
                # additional shape-1 check
                shape_len = len(inst.I)
                input_dict = None
                for _inputs, _ in success:
                    no_one = True
                    for i in range(shape_len):
                        if _inputs[i] == 1:
                            no_one = False
                            break
                    if no_one:
                        input_dict = list_to_dict(_inputs, "s")
                        break
                if input_dict != None:
                    for symb in inst.I:
                        origin_val = input_dict[symb]
                        input_dict[symb] = 1
                        _, outputs = input_validity_test(inst, deepcopy(input_dict))
                        if outputs == None:
                            valid = False
                            break
                        input_dict[symb] = origin_val
            # check whether the rule can reject all counterexamples
            if valid:
                for inputs in fail:
                    # if not all(map(lambda x : x >= 0, inputs)): continue
                    counterexample = False
                    input_dict = list_to_dict(inputs, "s")
                    concrete_input_tensors = abs_to_concrete(
                        abs_input_tensors, input_dict
                    )
                    try:
                        predicates = op.checked_requires(
                            list(
                                map(
                                    nnsmith_abstensor_from_ai_abstensor,
                                    concrete_input_tensors,
                                )
                            )
                        )
                    except IndexError:
                        counterexample = True
                    except ConstraintError:
                        counterexample = True
                    except InternalError:
                        counterexample = True
                    else:
                        for predicate in predicates:
                            if equivalent(predicate, False):
                                counterexample = True
                    if not counterexample:
                        valid = False
                        break
            if valid:
                res.append(ind)
                break
                # info[f'{inst.signature_hash}'].append(ind)
    with open(os.path.join(root_dir, f"{inst.name_index}.pkl"), "wb") as f:
        pickle.dump(res, f)
    with open("./nnsmith-process", "a") as f:
        f.write(inst.name + " " + str(opid) + " complete!\n")
    # print(inst.name, str(opid), "complete!", flush=True)


if __name__ == "__main__":
    gen_inst_records = gen_inst_with_records(
        data_dir=InvocDataDir,
        int_policy="fix_dim",
    )
    specify_nnsmith_list = [
        # '18107be68c229b00660960a1ec9059fc2427050fa5f94eaa3014eefaac9572f6',
        # '4be2ab8f70dc55d81fba6d2abf101cf655124dc9ae2aecec5a7384cf4cdb897b',
    ]
    os.system(f"mkdir -p {root_dir}")
    p = mp.Pool(parallel)
    for opid, (inst, records) in enumerate(gen_inst_records):
        if specify_nnsmith_list != [] and inst.name_index not in specify_nnsmith_list:
            continue
        p.apply_async(solve_inst, (opid, inst, records))
        # solve_inst(opid, inst, records)
    p.close()
    p.join()

    # with open(os.path.join(InformationRootDir, f'nnsmith_rule_table-{cfg_library}-{date}.pkl'), 'wb') as f:
    # pickle.dump(info, f)
