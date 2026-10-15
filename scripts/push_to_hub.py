"""Publish the rebuilt v1 model and the v1 Space to the Hugging Face Hub.

Usage (after build_v1.py, with a write token in HF_TOKEN or `hf auth login`):
    python scripts/push_to_hub.py --user makarios19
"""
import argparse
import os

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True, help="Hugging Face user or organization")
    p.add_argument("--model-name", default="IndoBerea-v1")
    p.add_argument("--space-name", default="IndoBerea-v1")
    p.add_argument("--build", default="build")
    args = p.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    model_id = f"{args.user}/{args.model_name}"
    space_id = f"{args.user}/{args.space_name}"

    api.create_repo(model_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=os.path.join(args.build, "model"), repo_id=model_id,
                      repo_type="model", commit_message="Upload rebuilt IndoBerea v1")
    print(f"model: https://huggingface.co/{model_id}")

    api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=os.path.join(ROOT, "space"), repo_id=space_id,
                      repo_type="space", commit_message="Upload Space app",
                      ignore_patterns=["__pycache__/*", "*.pyc"])
    api.upload_folder(folder_path=os.path.join(args.build, "space_index"), repo_id=space_id,
                      repo_type="space", path_in_repo="index", commit_message="Upload verse index")
    api.add_space_variable(space_id, "MODEL_ID", model_id)
    print(f"space: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
