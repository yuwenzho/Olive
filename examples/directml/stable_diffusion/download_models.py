import requests
import argparse
from sd_directories import get_directories
from pathlib import Path


if __name__ == "__main__":
    script_dir, models_dir, data_dir, cache_dir = get_directories()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=str(models_dir / "sd-unoptimized"),
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    )

    parser.add_argument(
        "--opset",
        default=14,
        type=int,
        help="The version of the ONNX operator set to use.",
    )

    args = parser.parse_args()

    # TODO: check if logged into huggingface cli?

    conversion_script_path = data_dir / "convert_sd.py"
    
    if not conversion_script_path.exists():
        response = requests.get('https://raw.githubusercontent.com/huggingface/diffusers/v0.13.0/scripts/convert_stable_diffusion_checkpoint_to_onnx.py')
        with open(conversion_script_path, "wb") as f:
            f.write(response.content)

    from data.convert_sd import convert_models

    convert_models(
        model_path=args.model_path,
        output_path=args.output_path,
        opset=args.opset
    )