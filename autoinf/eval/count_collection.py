import json
import logging
from os import PathLike
from typing import Dict, List, Tuple, Type

from autoinf.instrument.categorize import gen_inst_with_records
from autoinf.instrument.op import OpInstance

from dynofuzz.abstract.dtype import DType
from dynofuzz.abstract.tensor import AbsTensor

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
AUTOINF_LOG = logging.getLogger("autoinf")


def make_record_finder(
    gen_inst_records: List[Tuple[OpInstance, List[Tuple[Dict]]]] = None,
    path: PathLike = None,
    max_elem_per_tensor=2**16,
):
    if gen_inst_records is None:
        assert path is not None, "Either gen_inst_records or path must be provided."
        gen_inst_records = gen_inst_with_records(data_dir=path, int_policy="fix_dim")

    producer = {}
    consumer = {}
    inst2record = {}

    total_rec = 0
    skipped_elem = 0
    skipped_err = 0
    skipped_blacklist = 0

    blacklisted = set()

    api_set = set()

    for inst, records in gen_inst_records:
        total_rec += len(records)

        if inst.name in [
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
        ]:  # black list
            if inst.name not in blacklisted:
                AUTOINF_LOG.error(f"Blacklist operator {inst.name} found!")
                blacklisted.add(inst.name)
            skipped_blacklist += len(records)
            continue

        for record in records:
            try:
                input_abs_tensor = [
                    AbsTensor(shape, DType.from_str(inften.dtype))
                    for inften, shape in zip(
                        inst.input_tensors, inst.concrete_input_shapes(record[0])
                    )
                ]
                if any([x.nelement() > max_elem_per_tensor for x in input_abs_tensor]):
                    AUTOINF_LOG.debug(
                        f"Skip {inst.name} <- {input_abs_tensor} for over {max_elem_per_tensor} elements."
                    )
                    skipped_elem += 1
                    continue
            except Exception as e:
                AUTOINF_LOG.error(f"{inst.name}: bad input subst. {inst.I} -> {record}")
                skipped_err += 1
                continue

            try:
                output_abs_tensor = [
                    AbsTensor(shape, DType.from_str(inften.dtype))
                    for inften, shape in zip(
                        inst.output_tensors, inst.concrete_output_shapes(record[1])
                    )
                ]
                if any([x.nelement() > max_elem_per_tensor for x in output_abs_tensor]):
                    AUTOINF_LOG.debug(
                        f"Skip {inst.name} -> {output_abs_tensor} for over {max_elem_per_tensor} elements."
                    )
                    skipped_elem += 1
                    continue
            except Exception as e:
                AUTOINF_LOG.error(
                    f"{inst.name}: bad output subst. {inst.O} -> {record}"
                )
                skipped_err += 1
                continue

            for iten in input_abs_tensor:
                prod_list = producer.setdefault(iten, [])
                if inst not in prod_list:
                    prod_list.append(inst)

            for oten in output_abs_tensor:
                cons_list = consumer.setdefault(oten, [])
                if inst not in cons_list:
                    cons_list.append(inst)

            inst2record.setdefault(inst, []).append(
                (
                    tuple(input_abs_tensor),
                    tuple(output_abs_tensor),
                    {k: record[0][k] for k in inst.A},
                )
            )

            api_set.add(inst.name)
            # if inst.name in [
            #     'torch._C._fft.fft_ihfft',
            #     'torch._C._linalg.linalg_ldl_solve',
            # ]:
            #     ic(inst.name)
            #     embed()
        # end for record

    skipped_rec = skipped_elem + skipped_err + skipped_blacklist
    final_rec = total_rec - skipped_rec
    AUTOINF_LOG.info(
        f"Got {final_rec} records of {len(inst2record)} OpInstance of {len(api_set)} APIs"
    )
    AUTOINF_LOG.info(f"Filtered {skipped_rec} records from {total_rec} initial set.")
    AUTOINF_LOG.info(
        f"~ {skipped_elem}: over {max_elem_per_tensor} elem.  ~ {skipped_err}: bad subst.  ~ {skipped_blacklist}: blacklisted."
    )
    with open("eval/valid_apis.json", "w") as f:
        json.dump(list(api_set), indent=2, fp=f)

    inst_2_num_records = dict(
        sorted(
            {i.name_index: len(r) for i, r in inst2record.items()}.items(),
            key=lambda x: x[0],
        )
    )
    with open("eval/inst_2_num_records_from_collection.json", "w") as f:
        json.dump(inst_2_num_records, indent=2, fp=f)


make_record_finder(path="data/torch_records_1123_0_1")
