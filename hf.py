from huggingface_hub import HfApi, hf_hub_download
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

hf_api = HfApi()


def push_model(org_nm, project_nm, model_path, saved_model_dir="saved_model"):
    repo_id = f"{org_nm}/{project_nm}"
    hf_api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    hf_api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=f"{saved_model_dir}/checkpoint.pth",  # in huggingface
        repo_id=repo_id,
    )
    print("Pushed the model to HF hub successfully!!!")


def download_the_model(org_nm, project_nm, saved_model_dir="saved_model"):
    root_folder = os.getcwd()
    os.makedirs(saved_model_dir, exist_ok=True)
    repo_id = f"{org_nm}/{project_nm}"

    local_path = os.path.join(root_folder, saved_model_dir, "checkpoint.pth")
    if not os.path.exists(local_path):
        try:
            print("repo_id: ", repo_id)
            print("filename: ", f"{saved_model_dir}/checkpoint.pth")
            model_path = hf_hub_download(
                repo_id=repo_id, filename=f"{saved_model_dir}/checkpoint.pth"
            )

            shutil.move(model_path, local_path)
        except Exception as e:
            print(f"Failed to download or move model to {saved_model_dir}: {e}")

    print("Downloaded the model weights successfully!!!")
