"""GitHub Project Uploader using GitHub REST API.
Uploads the complete project directly to GitHub without requiring git.exe.
"""

import os
import sys
import base64
import argparse
import requests


def get_all_files(root_dir: str):
    """Recursively collects project files excluding build/cache artifacts."""
    exclude_dirs = {
        ".git", ".gemini", "__pycache__", "outputs", "venv", "env", ".venv",
        "node_modules", "data/cache", "experiments/logs"
    }
    exclude_extensions = {".pyc", ".pyo", ".pyd"}

    collected_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter directories in-place
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith(".")]

        rel_dir = os.path.relpath(dirpath, root_dir)
        if any(part in exclude_dirs for part in rel_dir.replace("\\", "/").split("/")):
            continue

        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in exclude_extensions:
                continue
            if f.endswith(".log") or f.endswith(".tmp"):
                continue

            full_path = os.path.join(dirpath, f)
            rel_path = os.path.relpath(full_path, root_dir).replace("\\", "/")
            collected_files.append((rel_path, full_path))

    return collected_files


def upload_to_github(repo_name: str, token: str, private: bool = False):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    # 1. Get authenticated user info
    print("[1/4] Authenticating with GitHub...")
    user_res = requests.get("https://api.github.com/user", headers=headers)
    if user_res.status_code != 200:
        print(f"Authentication failed ({user_res.status_code}): {user_res.text}")
        return False

    username = user_res.json().get("login")
    print(f"Authenticated as GitHub user: @{username}")

    # 2. Check if repository exists or create it
    print(f"[2/4] Checking repository '{username}/{repo_name}'...")
    repo_check = requests.get(f"https://api.github.com/repos/{username}/{repo_name}", headers=headers)

    if repo_check.status_code == 404:
        print(f"Creating new GitHub repository '{repo_name}'...")
        create_payload = {
            "name": repo_name,
            "description": "Chandrayaan-2 Optical Image Correspondence and Registration Pipeline (SIH 26166)",
            "private": private,
            "auto_init": True
        }
        create_res = requests.post("https://api.github.com/user/repos", headers=headers, json=create_payload)
        if create_res.status_code not in [200, 201]:
            print(f"Failed to create repo: {create_res.text}")
            return False
        print("Repository successfully created!")
    else:
        print("Repository already exists. Proceeding to upload/update files...")

    # 3. Gather project files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files_to_upload = get_all_files(current_dir)
    print(f"[3/4] Prepared {len(files_to_upload)} files to upload.")

    # 4. Upload each file using the GitHub Contents API
    print("[4/4] Uploading files to GitHub...")
    for i, (rel_path, full_path) in enumerate(files_to_upload, 1):
        try:
            with open(full_path, "rb") as f:
                content_bytes = f.read()

            b64_content = base64.b64encode(content_bytes).decode("utf-8")

            # Check if file already exists in repo to get SHA
            sha = None
            check_url = f"https://api.github.com/repos/{username}/{repo_name}/contents/{rel_path}"
            existing = requests.get(check_url, headers=headers)
            if existing.status_code == 200:
                sha = existing.json().get("sha")

            payload = {
                "message": f"Upload {rel_path} for Chandrayaan-2 correspondence pipeline",
                "content": b64_content,
                "branch": "main"
            }
            if sha:
                payload["sha"] = sha

            put_res = requests.put(check_url, headers=headers, json=payload)
            if put_res.status_code in [200, 201]:
                print(f"  [{i}/{len(files_to_upload)}] Uploaded: {rel_path}")
            else:
                print(f"  [{i}/{len(files_to_upload)}] Warning on {rel_path}: {put_res.status_code} - {put_res.text[:80]}")

        except Exception as e:
            print(f"  [{i}/{len(files_to_upload)}] Error uploading {rel_path}: {e}")

    repo_url = f"https://github.com/{username}/{repo_name}"
    print("\n" + "=" * 70)
    print("SUCCESS! Your project has been uploaded to GitHub:")
    print(f"👉 {repo_url}")
    print("=" * 70)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload project to GitHub")
    parser.add_argument("--repo", type=str, default="chandrayaan2-optical-correspondence",
                        help="Repository name on GitHub")
    parser.add_argument("--token", type=str, default=None,
                        help="GitHub Personal Access Token (classic or fine-grained with 'repo' permission)")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    args = parser.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN")

    if not token:
        print("GitHub Project Uploader")
        print("-" * 50)
        token = input("Enter your GitHub Personal Access Token (PAT): ").strip()

    if not token:
        print("Error: GitHub Token is required.")
        sys.exit(1)

    upload_to_github(repo_name=args.repo, token=token, private=args.private)

