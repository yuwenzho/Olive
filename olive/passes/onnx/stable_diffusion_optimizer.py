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
from onnxruntime.transformers.fusion_options import FusionOptions  # noqa: E402
import onnxruntime as ort
from onnxruntime.transformers.optimizer import optimize_model  # noqa: E402


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
                type_=bool, default=True, description="Whether half-precision float will be used."
            ),
            "force_fp32_ops": PassConfigParam(
                type_=list[str], default=[], description="Operators that are forced to run in float32"
            ),
            "use_external_data_format": PassConfigParam(
                type_=bool, default=False, description="Whether use external data format to store large model (>2GB)"
            ),
        }


    def _run_for_config(self, model: ONNXModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
        if Path(output_model_path).suffix != ".onnx":
            output_model_path += ".onnx"

        config = self._config_class(**config)

        fusion_options = FusionOptions(config.model_type)
        # TODO: equivalent of fusion_options.parse(args) to add additional options from config

        if config.model_type == "unet":
            # Some optimizations are not available in v1.14 or older version: packed QKV and BiasAdd
            has_all_optimizations = version.parse(ort.__version__) >= version.parse("1.15.0")
            fusion_options.enable_packed_kv = config.float16
            fusion_options.enable_packed_qkv = config.float16 and has_all_optimizations
            fusion_options.enable_bias_add = has_all_optimizations

        m = optimize_model(
            str(model.model_path),
            model_type=config.model_type,
            num_heads=0,  # will be deduced from graph
            hidden_size=0,  # will be deduced from graph
            opt_level=0,  # TODO set this?
            optimization_options=fusion_options,
            use_gpu=True,  # TODO if "cuda" or "dml" EP
        )

        if config.float16:
            op_block_list = ['RandomNormalLike'] + config.force_fp32_ops
            m.convert_float_to_float16(keep_io_types=False, op_block_list=op_block_list)

        m.save_model_to_file(str(output_model_path), use_external_data_format=False)

        return ONNXModel(output_model_path, model.name)
