import json
import os

from autoinf.inference.api_parsing import OpDatabase
from autoinf.inference.configs import *


def count_aug():
    black_list = [
        "torch.gather",
        "torch.nanquantile",
        "torch.Tensor.resize_as_",  # resize_as_ can't be represented in the JIT at the moment ...
        "torch.Tensor.rename",
        "torch.Tensor.rename_",
        "torch.Tensor.uniform_",
        "torch.Tensor.normal_",
        "torch.Tensor.exponential_",
        "torch.Tensor.poisson_",
        "torch.Tensor.geometric_",
        "torch.Tensor.log_normal_",
        "torch.Tensor.cauchy_",
        "torch.Tensor.logistic_",
        "torch.Tensor.random_",
        "torch.Tensor.requires_grad_",
        "torch.nn.functional.dropout",
        "torch.nn.functional.dropout2d",
        "torch.nn.functional.dropout3d",
    ]
    valid_apis = set(json.load(open("eval/valid_apis.json")))
    skipped_apis = set()
    filelist = os.listdir(InvocDBDir)
    filelist = list(filter(lambda x: not any(x in b for b in black_list), filelist))

    api_2_num_inst = {}
    inst_2_num_records = {}

    for filename in filelist:
        api_name = filename.split("-")[0]
        if api_name not in valid_apis:
            if api_name not in skipped_apis:
                print(f"{api_name} not in valid_apis.json, skipping...")
                skipped_apis.add(api_name)
            continue

        num_inst = api_2_num_inst.get(api_name, 0)
        api_2_num_inst[api_name] = num_inst + 1

        file_path = os.path.join(InvocDBDir, filename)
        OpDB = OpDatabase(file_path)
        num_records = OpDB.InvocationCount(type="success")  # after augmentation
        inst_2_num_records[filename] = num_records
    # end for

    for api in valid_apis:
        if api not in api_2_num_inst:
            print(f"Warning: Not augmented API: {api}")

    inst_2_num_records = dict(sorted(inst_2_num_records.items(), key=lambda x: x[0]))
    with open("eval/inst_2_num_records_after_aug.json", "w") as f:
        json.dump(inst_2_num_records, indent=2, fp=f)


if __name__ == "__main__":
    count_aug()
