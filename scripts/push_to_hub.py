"""Publish IndoBerea v1 to the Hugging Face Hub.

Parts (choose with --what):
    model  the rebuilt model in build/model (run build_v1.py first)
    card   only the model card (model_card/README.md), e.g. after editing it
    space  the static Space in space/ (free; no model runs in it)

Usage (with a write token in HF_TOKEN or `hf auth login`):
    python scripts/push_to_hub.py --user makarios19 --what card space
"""
import argparse
import os
import subprocess
import sys

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True, help="Hugging Face user or organization")
    p.add_argument("--what", nargs="+", choices=["model", "card", "space"], default=["card", "space"])
    p.add_argument("--model-name", default="IndoBerea-v1")
    p.add_argument("--space-name", default="IndoBerea-v1")
    p.add_argument("--build", default="build")
    args = p.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    model_id = f"{args.user}/{args.model_name}"
    space_id = f"{args.user}/{args.space_name}"

    if "model" in args.what:
        api.create_repo(model_id, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=os.path.join(args.build, "model"), repo_id=model_id,
                          repo_type="model", commit_message="Upload rebuilt IndoBerea v1")
        print(f"model: https://huggingface.co/{model_id}")

    if "card" in args.what:
        api.upload_file(path_or_fileobj=os.path.join(ROOT, "model_card", "README.md"),
                        path_in_repo="README.md", repo_id=model_id, repo_type="model",
                        commit_message="Update model card")
        print(f"model card: https://huggingface.co/{model_id}")

    if "space" in args.what:
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_space_data.py")], check=True)
        api.create_repo(space_id, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(folder_path=os.path.join(ROOT, "space"), repo_id=space_id,
                          repo_type="space", commit_message="Upload static Space")
        print(f"space: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
