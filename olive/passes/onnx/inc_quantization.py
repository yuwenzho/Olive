# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Union

from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam
from olive.strategy.search_parameter import Boolean, Categorical, Conditional

logger = logging.getLogger(__name__)

_inc_quantization_config = {
    "device": PassConfigParam(
        type_=str,
        default_value="cpu",
        description="""
            Intel® Neural Compressor quantization device. Support 'cpu' and 'gpu'.
        """,
    ),
    "backend": PassConfigParam(
        type_=str,
        default_value="default",
        description="""
            Backend for model execution. Support 'default', 'onnxrt_trt_ep', 'onnxrt_cuda_ep'
        """,
    ),
    "domain": PassConfigParam(
        type_=str,
        default_value="auto",
        description="""
            Model domain. Support 'auto', 'cv', 'object_detection', 'nlp' and 'recommendation_system'.
            Intel® Neural Compressor Adaptor will use specific quantization settings for different domains
            automatically, and explicitly specified quantization settings will override the automatic setting.
            If users set domain as auto, automatic detection for domain will be executed.
        """,
    ),
    "recipes": PassConfigParam(
        type_=dict,
        default_value={},
        description="""
            Recipes for Intel® Neural Compressor quantiztaion, support list is as below.
                'smooth_quant': whether do smooth quant
                'smooth_quant_args': parameters for smooth_quant
                'fast_bias_correction': whether do fast bias correction
                'weight_correction': whether do weight correction
                'gemm_to_matmul': whether convert gemm to matmul and add, only valid for onnx models
                'graph_optimization_level': support 'DISABLE_ALL', 'ENABLE_BASIC', 'ENABLE_EXTENDED', 'ENABLE_ALL'
                                        only valid for onnx models
                'first_conv_or_matmul_quantization': whether quantize the first conv or matmul
                'last_conv_or_matmul_quantization': whether quantize the last conv or matmul
                'pre_post_process_quantization': whether quantize the ops in preprocess and postprocess
                'add_qdq_pair_to_weight': whether add QDQ pair for weights, only vaild for onnxrt_trt_ep
                'optypes_to_exclude_output_quant': don't quantize output of specified optypes
                'dedicated_qdq_pair': whether dedicate QDQ pair, only vaild for onnxrt_trt_ep
        """,
    ),
    "reduce_range": PassConfigParam(
        type_=bool,
        default_value=False,
        searchable_values=Boolean(),
        description="""
            Whether use 7 bit to quantization.
        """,
    ),
    "quant_level": PassConfigParam(
        type_=str,
        default_value="auto",
        description="""
            Intel® Neural Compressor allows users to choose different tuning processes by specifying
            the quantization level (quant_level). Currently 3 quant_levels are supported.
            0 is conservative strategy, 1 is basic or user-specified strategy,
            auto (default) is the combination of 0 and 1.
            Please refer to
            https://github.com/intel/neural-compressor/blob/master/docs/source/tuning_strategies.md#tuning-process
            https://github.com/intel/neural-compressor/blob/master/docs/source/tuning_strategies.md#tuning-algorithms
            for more details
        """,
    ),
    "excluded_precisions": PassConfigParam(
        type_=list,
        default_value=[],
        description="""
            Precisions to be excluded, Default value is empty list.
            Intel® Neural Compressor enable the mixed precision with
            fp32 + bf16(only when device is 'gpu' and backend is 'onnxrt_cuda_ep') + int8 by default.
            If you want to disable bf16 data type, you can specify excluded_precisions = ['bf16'].
        """,
    ),
    "accuracy_criterion": PassConfigParam(
        type_=dict,
        default_value={},
        description="""
            Accuracy constraint settings.
        """,
    ),
    "tuning_criterion": PassConfigParam(
        type_=dict,
        default_value={},
        description="""
            Instance of TuningCriterion class. In this class you can set strategy, strategy_kwargs,
            timeout, max_trials and objective.
        """,
    ),
    "evaluate_func": PassConfigParam(
        type_=Union[Callable, str],
        is_object=True,
        default_value=None,
        description="""
            Evaluation function.
        """,
    ),
}

_inc_static_dataloader_config = {
    "data_dir": PassConfigParam(
        type_=Union[Path, str],
        is_path=True,
        description="""
            Path to the directory containing the dataset.
            For local data, it is required if approach is 'static'.
        """,
    ),
    "batch_size": PassConfigParam(
        type_=int,
        default_value=1,
        description="""
            Batch size for calibration, required if approach is 'static'.
        """,
    ),
    "dataloader_func": PassConfigParam(
        type_=Union[Callable, str],
        required=True,
        is_object=True,
        description="""
            Function/function name to generate dataloader for calibration,
            required if approach is 'static'
        """,
    ),
}

_inc_static_optional_config = {
    "quant_format": PassConfigParam(
        type_=str,
        default_value="QOperator",
        searchable_values=Categorical(["QOperator", "QDQ"]),
        description="""
            Quantization format. Support 'QDQ' and 'QOperator'.
        """,
    ),
    "calibration_sampling_size": PassConfigParam(
        type_=Union[list, int],
        default_value=[100],
        description="""
            Number of calibration sample.
        """,
    ),
}

_inc_accuracy_criterion_config = {
    "higher_is_better": PassConfigParam(
        type_=bool,
        default_value=True,
        description="""
            This flag indicates whether the metric higher is the better.
            Default value is True.
        """,
    ),
    "criterion": PassConfigParam(
        type_=str,
        default_value="relative",
        description="""
            This flag indicates whether the metric loss is 'relative' or 'absolute'.
            Default value is 'relative'.
        """,
    ),
    "tolerable_loss": PassConfigParam(
        type_=float,
        default_value=0.01,
        description="""
            This float indicates how much metric loss we can accept.
            Default value is 0.01.
        """,
    ),
}

_inc_tuning_criterion_config = {
    "strategy": PassConfigParam(
        type_=str,
        default_value="basic",
        description="""
            Strategy name used in tuning. Details in
            https://github.com/intel/neural-compressor/blob/master/docs/source/tuning_strategies.md#basic
        """,
    ),
    "strategy_kwargs": PassConfigParam(
        type_=dict,
        default_value=None,
        description="""
            Parameters for strategy.
        """,
    ),
    "timeout": PassConfigParam(
        type_=int,
        default_value=0,
        description="""
            Tuning timeout (seconds). Default value is 0 which means early stop.
        """,
    ),
    "max_trials": PassConfigParam(
        type_=int,
        default_value=5,
        description="""
            Max tune times. Default value is 5. Combine with timeout field to decide when to exit.
        """,
    ),
    "objective": PassConfigParam(
        type_=str,
        default_value="performance",
        description="""
            String or dict. Objective with accuracy constraint guaranteed. String value supports
            'performance', 'modelsize', 'footprint'. Default value is 'performance'.
        """,
    ),
}


class IncQuantization(Pass):
    """
    Quantize ONNX model with Intel® Neural Compressor.
    """

    _requires_user_script = True
    _requires_data_config = True

    def _initialize(self):
        super()._initialize()

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        config = {
            "approach": PassConfigParam(
                type_=str,
                default_value="static",
                searchable_values=Categorical(["dynamic", "static"]),
                description="""
                Intel® Neural Compressor Quantization mode. 'dynamic' for dynamic quantization,
                'static' for static quantization.
            """,
            )
        }

        # common quantization config
        config.update(deepcopy(_inc_quantization_config))

        # accuracy criterion and tuning criterion config
        for key, value in deepcopy(_inc_accuracy_criterion_config).items():
            config["accuracy_criterion"].default_value.update({key: value.default_value})
        for key, value in deepcopy(_inc_tuning_criterion_config).items():
            config["tuning_criterion"].default_value.update({key: value.default_value})

        # static quantization config
        config.update(deepcopy(_inc_static_dataloader_config))
        inc_static_optional_config = deepcopy(_inc_static_optional_config)
        for _, value in inc_static_optional_config.items():
            # default value of quant_format is conditional on approach
            if isinstance(value.searchable_values, Categorical):
                # ignore the parameter quant_format if approach is dynamic, if approach is static,
                # use the searchable_values in inc_static_optional_config by making it conditional
                value.searchable_values = Conditional(
                    parents=("approach",),
                    support={("static",): value.searchable_values},
                    default=Categorical(["default"]),
                )
            elif isinstance(value.searchable_values, Conditional):
                # ignore the parameter quant_format if approach is dynamic, if approach is static,
                # use the searchable_values in inc_static_optional_config by expanding the parents
                value.searchable_values = Conditional(
                    parents=("approach",) + value.searchable_values.parents,
                    support={
                        ("static",) + key: value.searchable_values.support[key]
                        for key in value.searchable_values.support
                    },
                    default=Categorical(["default"]),
                )
        config.update(inc_static_optional_config)

        # external data config
        config.update(get_external_data_config())
        return config

    def _run_for_config(self, model: ONNXModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
        try:
            from neural_compressor import quantization
            from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
        except ImportError:
            raise ImportError(
                "Please install `olive-ai[inc]` or `neural-compressor` to use Intel® Neural Compressor quantization"
            )

        # start with a copy of the config
        run_config = deepcopy(config)
        is_static = run_config["approach"] == "static"

        output_model_path = ONNXModel.resolve_path(output_model_path)

        # keys not needed for quantization
        accuracy_criterion = AccuracyCriterion(**run_config["accuracy_criterion"])
        tuning_criterion = TuningCriterion(**run_config["tuning_criterion"])
        to_delete = [
            "script_dir",
            "user_script",
            "data_dir",
            "batch_size",
            "dataloader_func",
            "accuracy_criterion",
            "tuning_criterion",
            "evaluate_func",
            "data_config",
        ]
        to_delete += list(get_external_data_config().keys())
        for key in to_delete:
            if key in run_config:
                del run_config[key]

        ptq_config = PostTrainingQuantConfig(
            **run_config, accuracy_criterion=accuracy_criterion, tuning_criterion=tuning_criterion
        )
        inc_calib_dataloader = None
        if is_static:
            if self._user_module_loader:
                inc_calib_dataloader = self._user_module_loader.call_object(
                    self._fixed_params["dataloader_func"],
                    self._fixed_params["data_dir"],
                    self._fixed_params["batch_size"],
                )
            elif self._data_config:
                inc_calib_dataloader = self._data_config.to_data_container().create_calibration_dataloader()

        eval_func = (
            self._user_module_loader.load_object(self._fixed_params["evaluate_func"])
            if "evaluate_func" in self._fixed_params
            else None
        )

        q_model = quantization.fit(
            model.model_path, ptq_config, calib_dataloader=inc_calib_dataloader, eval_func=eval_func
        )

        # save the model to the output path and return the model
        return model_proto_to_olive_model(q_model.model, output_model_path, config, model.name)


class IncDynamicQuantization(IncQuantization):
    """Intel® Neural Compressor Dynamic Quantization Pass"""

    _requires_user_script = True

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        config = {
            "approach": PassConfigParam(type_=str, default_value="dynamic", description="dynamic quantization mode")
        }
        # common quantization config
        config.update(deepcopy(_inc_quantization_config))

        # accuracy criterion and tuning criterion config
        for key, value in deepcopy(_inc_accuracy_criterion_config).items():
            config["accuracy_criterion"].default_value.update({key: value.default_value})
        for key, value in deepcopy(_inc_tuning_criterion_config).items():
            config["tuning_criterion"].default_value.update({key: value.default_value})

        # external data config
        config.update(get_external_data_config())
        return config


class IncStaticQuantization(IncQuantization):
    """Intel® Neural Compressor Static Quantization Pass"""

    _requires_user_script = True

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        config = {
            "approach": PassConfigParam(type_=str, default_value="static", description="static quantization mode")
        }
        # common quantization config
        config.update(deepcopy(_inc_quantization_config))

        # accuracy criterion and tuning criterion config
        for key, value in deepcopy(_inc_accuracy_criterion_config).items():
            config["accuracy_criterion"].default_value.update({key: value.default_value})
        for key, value in deepcopy(_inc_tuning_criterion_config).items():
            config["tuning_criterion"].default_value.update({key: value.default_value})

        # static quantization specific config
        config.update(deepcopy(_inc_static_dataloader_config))
        config.update(deepcopy(_inc_static_optional_config))
        # external data config
        config.update(get_external_data_config())
        return config
