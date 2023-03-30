# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from packaging import version

from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam


class OnnxStableDiffusionOptimization(Pass):
    """Optimize stable diffusion models in scenarios where ONNX Runtime does not apply the optimization at load time.
    It is based on onnxruntime.transformers.optimizer."""

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        return {
            "model_type": PassConfigParam(
                type_=str,
                required=True,
                description=("One of unet, vae, clip, safety_checker."),
            ),
            # TODO: make this a search param (some HW may not run faster with fp16)
            "float16": PassConfigParam(
                type_=bool, default=False, description="Whether half-precision float will be used."
            ),
            # TODO
            # "force_fp32_ops": PassConfigParam(
            #     type_=list[str], default=[], description="Operators that are forced to run in float32"
            # ),
            "use_external_data_format": PassConfigParam(
                type_=bool, default=False, description="Whether use external data format to store large model (>2GB)"
            ),
        }

    def _run_for_config(self, model: ONNXModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
        import onnxruntime as ort

        if version.parse(ort.__version__) < version.parse("1.15.0"):
            raise RuntimeError("This pass requires onnxruntime 1.15.0 or newer")

        # TODO: implement. This is a passthrough right now
        import shutil
        shutil.copyfile(str(model.model_path), output_model_path)

        # from onnxruntime.transformers import optimizer as transformers_optimizer

        # from onnxruntime.transformers.fusion_options import FusionOptions
        # from onnxruntime.transformers.onnx_model_clip import ClipOnnxModel

        # sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
        # from fusion_options import FusionOptions  # noqa: E402
        # from onnx_model_clip import ClipOnnxModel  # noqa: E402
        # from onnx_model_unet import UnetOnnxModel  # noqa: E402
        # from onnx_model_vae import VaeOnnxModel  # noqa: E402
        # from optimizer import optimize_by_onnxruntime, optimize_model  # noqa: E402

        # # start with a copy of the config
        # run_config = deepcopy(config)
        # del run_config["float16"], run_config["input_int32"], run_config["use_external_data_format"]

        # optimizer = transformers_optimizer.optimize_model(input=model.model_path, **run_config)
        # if config["float16"]:
        #     optimizer.convert_float_to_float16(keep_io_types=True)
        # if config["input_int32"]:
        #     optimizer.change_graph_inputs_to_int32()

        # # add onnx extension if not present
        # if Path(output_model_path).suffix != ".onnx":
        #     output_model_path += ".onnx"

        # optimizer.save_model_to_file(output_model_path, config["use_external_data_format"])

        return ONNXModel(output_model_path, model.name)
