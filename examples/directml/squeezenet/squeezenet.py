# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import warnings
import requests

from olive.engine import Engine
from olive.evaluator.metric import LatencySubType, Metric, MetricType
from olive.evaluator.olive_evaluator import OliveEvaluator
from olive.model import ONNXModel
from olive.passes import OnnxStableDiffusionOptimization, OrtPerfTuning
from olive.systems.local import LocalSystem
from pathlib import Path
import onnxruntime as ort

warnings.simplefilter(action="ignore", category=FutureWarning)

ort.set_default_logger_severity(3)


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    models_dir = current_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = current_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Input Model
    input_model_path = models_dir / "squeezenet1.1-7.onnx"
    if not input_model_path.exists():
        response = requests.get('https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.1-7.onnx')
        with open(input_model_path, "wb") as f:
            f.write(response.content)

    input_model = ONNXModel(str(input_model_path), is_file=True)

    # ------------------------------------------------------------------
    # Evaluator
    latency_metric_config = {
        "user_script": str(current_dir / "user_script.py"),
        "dataloader_func": "create_dataloader",
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
            "execution_order": "pass-by-pass",
            "search_algorithm": "exhaustive",
        },
        "clean_cache": True,  # TODO: remove
    }
    engine = Engine(options, evaluator=evaluator)

    # ------------------------------------------------------------------
    # ONNX Runtime performance tuning pass
    ort_perf_tuning_config = {
        "user_script": str(current_dir / "user_script.py"),
        "dataloader_func": "create_dataloader",
        "batch_size": 1,
        "device": "cpu",
        # "providers_list": ["DmlExecutionProvider"],
    }
    ort_perf_tuning_pass = OrtPerfTuning(ort_perf_tuning_config)
    engine.register(ort_perf_tuning_pass)

    # ------------------------------------------------------------------
    # Run engine
    best_execution = engine.run(input_model, verbose=True)
    print(best_execution)