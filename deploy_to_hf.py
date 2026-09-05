"""Automated Hugging Face Spaces Deployment Script.
Creates a Streamlit Space on Hugging Face and uploads all required project files.
"""

import os
import sys
import argparse
from huggingface_hub import HfApi, create_repo, upload_folder


def deploy(repo_id: str, token: str = None):
    """
    Deploys current directory to Hugging Face Spaces.
    Args:
        repo_id: e.g. "username/chandrayaan2-correspondence"
        token: Hugging Face User Access Token (with write permission)
    """
    api = HfApi(token=token)

    print(f"[HF DEPLOY] Authenticating and creating Space: {repo_id}...")
    try:
        url = create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="streamlit",
            token=token,
            exist_ok=True,
            private=False
        )
        print(f"[HF DEPLOY] Space created / verified: {url}")
    except Exception as e:
        print(f"[HF DEPLOY] Notice during repo creation: {e}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[HF DEPLOY] Uploading repository files to Hugging Face Space...")

    # Upload the folder, ignoring unneeded cache and git artifacts
    ignore_patterns = [
        "*.pyc",
        "__pycache__/*",
        ".git/*",
        ".gemini/*",
        "outputs/*",
        "experiments/logs/*",
        ".venv/*",
        "env/*"
    ]

    upload_folder(
        folder_path=current_dir,
        repo_id=repo_id,
        repo_type="space",
        token=token,
        ignore_patterns=ignore_patterns
    )

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    print("\n" + "=" * 70)
    print("🚀 SUCCESS! Your Space has been deployed to Hugging Face:")
    print(f"👉 {space_url}")
    print("=" * 70)
    print("Hugging Face will now automatically build and launch the container.")
    print("You can monitor the build log on the Space page.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy to Hugging Face Spaces")
    parser.add_argument("--repo", type=str, help="HF Space repo_id (e.g. your-username/chandrayaan2-pipeline)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Write Access Token")
    args = parser.parse_args()

    repo = args.repo
    token = args.token or os.environ.get("HF_TOKEN")

    if not repo:
        print("Hugging Face Spaces Deployment Helper")
        print("-" * 40)
        repo = input("Enter your Hugging Face Space ID (e.g. 'username/chandrayaan2-pipeline'): ").strip()

    if not token:
        token = input("Enter your Hugging Face Write Token (get it from https://huggingface.co/settings/tokens): ").strip()

    if not repo or not token:
        print("Error: Both Repo ID and Token are required.")
        sys.exit(1)

    deploy(repo_id=repo, token=token)

