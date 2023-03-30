# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import warnings

import onnxruntime as ort
from sd_directories import get_directories

from olive.engine import Engine
from olive.evaluator.metric import AccuracySubType, LatencySubType, Metric, MetricType
from olive.evaluator.olive_evaluator import OliveEvaluator
from olive.model import ONNXModel
from olive.passes import OnnxStableDiffusionOptimization, OrtPerfTuning
from olive.systems.local import LocalSystem
from pathlib import Path

warnings.simplefilter(action="ignore", category=FutureWarning)

ort.set_default_logger_severity(3)


def get_args():
    parser = argparse.ArgumentParser(description="Olive stable diffusion example")
    parser.add_argument(
        "--models_path",
        type=str,
        default=str(),
        help="Metric to optimize for: accuracy or latency",
    )
    parser.add_argument(
        "--search_algorithm",
        type=str,
        choices=["exhaustive", "random", "tpe"],
        default="exhaustive",
        help="Search algorithm: exhaustive or random",
    )
    parser.add_argument(
        "--execution_order",
        type=str,
        choices=["joint", "pass-by-pass"],
        default="pass-by-pass",
        help="Execution order: joint or pass-by-pass",
    )
    args = parser.parse_args()
    return args


def optimize_sd_model(unoptimized_model_path: Path):
    # directories
    current_dir, models_dir, _, cache_dir = get_directories()
    user_script = str(current_dir / "user_script.py")
    name = "resnet_trained_for_cifar10"

    # ------------------------------------------------------------------
    # Evaluator
    latency_metric_config = {
        "user_script": user_script,
        "dataloader_func": "create_benchmark_dataloader",
        "batch_size": 1,
    }
    latency_metric = Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_type=LatencySubType.AVG,
        higher_is_better=False,
        user_config=latency_metric_config,
    )
    evaluator = OliveEvaluator(metrics=[latency_metric], target=LocalSystem())

    # ------------------------------------------------------------------
    # Engine
    options = {
        "cache_dir": str(cache_dir),
        "search_strategy": {
            "execution_order": args.execution_order,
            "search_algorithm": args.search_algorithm,
        },
    }
    engine = Engine(options, evaluator=evaluator)

    # ------------------------------------------------------------------
    # Stable Diffusion optimization pass
    sd_config = {
        "model_type": "unet",
        "float16": True,
        # "use_external_data_format": True,
    }
    sd_pass = OnnxStableDiffusionOptimization(sd_config, disable_search=True)
    engine.register(sd_pass)

    # ------------------------------------------------------------------
    # ONNX Runtime performance tuning pass
    ort_perf_tuning_config = {
        "user_script": user_script,
        "dataloader_func": "create_benchmark_dataloader",
        "batch_size": 1,
    }
    ort_perf_tuning_pass = OrtPerfTuning(ort_perf_tuning_config)
    engine.register(ort_perf_tuning_pass)

    # ------------------------------------------------------------------
    # Input model
    input_model = ONNXModel(str(unoptimized_model_path), is_file=True)

    # ------------------------------------------------------------------
    # Run engine
    best_execution = engine.run(input_model, verbose=True)
    print(best_execution)

    return best_execution["metric"]


if __name__ == "__main__":
    args = get_args()

    # TODO: loop over all models
    optimize_sd_model(Path(args.models_path) / "unet" / "model.onnx")
