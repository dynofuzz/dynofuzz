import os
import pickle
from collections import defaultdict

from autoinf.inference.configs import *
from autoinf.inference.tree import gen_tree_from_string


def gen_type_transfer_trees(inst):
    rule_filename = f"{inst.name_index}.pkl"
    lib = "torch" if "torch" in inst.name_index else "tf"
    type_transfer_rules = defaultdict(list)
    type_transfer_dbg_info = ""
    type_transfer_rule_dir = os.path.join(
        InformationRootDir, f"{lib}_rules/IO_rules"
    )
    try:
        with open(os.path.join(type_transfer_rule_dir, rule_filename), "rb") as f:
            type_transfer_info = pickle.load(f)
        for o_id in range(type_transfer_info["output_rank"]):
            info_rules = type_transfer_info["output_rules"][o_id]
            for rule in info_rules["rules"][:3]:
                type_transfer_rules[f"o{o_id}"].append(gen_tree_from_string(rule))
                type_transfer_dbg_info += f"o{o_id}: {rule}\n"
    except:
        type_transfer_dbg_info += "no inferred rules\n"
    return type_transfer_rules, type_transfer_dbg_info


def gen_requires_trees(inst):
    rule_filename = f"{inst.name_index}.pkl"
    lib = "torch" if "torch" in inst.name_index else "tf"
    requires_rules = []
    requires_dbg_info = ""
    input_constraint_rule_dir = os.path.join(
        InformationRootDir, f"{lib}_rules/input_rules"
    )
    try:
        with open(os.path.join(input_constraint_rule_dir, rule_filename), "rb") as f:
            input_constraint_info = pickle.load(f)
        for (rule, sign) in input_constraint_info["rules"]:
            requires_rules.append((gen_tree_from_string(rule), sign))
            requires_dbg_info += f"{rule} {sign} 0\n"
    except:
        requires_dbg_info += "no inferred rules\n"
    return requires_rules, requires_dbg_info


def gen_nnsmith_rules(inst):
    lib = "torch" if "torch" in inst.name_index else "tf"
    try:
        with open(
            os.path.join(
                InformationRootDir,
                f"{lib}_rules/nnsmith_rules",
                f"{inst.name_index}.pkl",
            ),
            "rb",
        ) as f:
            res = pickle.load(f)
    except:
        res = []
    return res


def shape_transfer_valid(inst) -> bool:
    lib = "torch" if "torch" in inst.name_index else "tf"
    judge_result_dir = os.path.join(InformationRootDir, f"{lib}_rules/rule_validity")
    try:
        with open(os.path.join(judge_result_dir, f"{inst.name_index}.pkl"), "rb") as f:
            valid = pickle.load(f)[1]
    except:
        valid = False
    return valid


def constraint_valid(inst) -> bool:
    lib = "torch" if "torch" in inst.name_index else "tf"
    judge_result_dir = os.path.join(InformationRootDir, f"{lib}_rules/rule_validity")
    try:
        with open(os.path.join(judge_result_dir, f"{inst.name_index}.pkl"), "rb") as f:
            valid = pickle.load(f)[2]
    except:
        valid = False
    return valid


def infer_failure(inst) -> bool:
    lib = "torch" if "torch" in inst.name_index else "tf"
    judge_result_dir = os.path.join(InformationRootDir, f"{lib}_rules/rule_validity")
    try:
        with open(os.path.join(judge_result_dir, f"{inst.name_index}.pkl"), "rb") as f:
            valid = pickle.load(f)[0]
    except:
        valid = False
    return False if valid else True


def judge_failure(inst) -> bool:
    if len(inst.nnsmith_rules_list) > 0:
        return False
    return infer_failure(inst)
