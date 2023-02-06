import GPUtil
import pytest

if not GPUtil.getAvailable():
    pytest.skip(
        "Skipping TensorRT tests due to no GPU detected.", allow_module_level=True
    )

from dynofuzz.abstract.dtype import DType
from dynofuzz.backends import BackendFactory
from dynofuzz.graph_gen import model_gen
from dynofuzz.materialize import Model, TestCase
from dynofuzz.narrow_spec import auto_opconfig, auto_opset

TestCase.__test__ = False  # supress PyTest warning


def test_narrow_spec_cache_make_and_reload():
    factory = BackendFactory.init("tensorrt", target="cuda", optmax=True)
    ONNXModel = Model.init("onnx")
    opset_lhs = auto_opconfig(ONNXModel, factory)
    assert opset_lhs, "Should not be empty... Something must go wrong."
    opset_rhs = auto_opconfig(ONNXModel, factory)
    assert opset_lhs == opset_rhs

    # Assert types
    assert isinstance(opset_lhs["core.ReLU"].in_dtypes[0][0], DType)

    # Assert Dictionary Type Equality
    assert type(opset_lhs) == type(opset_rhs)
    assert type(opset_lhs["core.ReLU"]) == type(opset_rhs["core.ReLU"])
    assert type(opset_lhs["core.ReLU"].in_dtypes[0][0]) == type(
        opset_rhs["core.ReLU"].in_dtypes[0][0]
    )


def test_synthesized_onnx_model(tmp_path):
    d = tmp_path / "test_trt_onnx"
    d.mkdir()

    ONNXModel = Model.init("onnx")
    factory = BackendFactory.init("tensorrt", target="cuda", optmax=True)

    gen = model_gen(
        opset=auto_opset(ONNXModel, factory),
        seed=23132,
        max_nodes=1,
    )  # One op should not be easily wrong... I guess.

    model = ONNXModel.from_gir(gen.make_concrete())

    assert model.with_torch

    model.refine_weights()  # either random generated or gradient-based.
    oracle = model.make_oracle()

    testcase = TestCase(model, oracle)
    testcase.dump(root_folder=d)

    assert factory.verify_testcase(testcase) is None
