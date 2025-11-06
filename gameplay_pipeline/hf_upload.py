"""Upload files to Hugging Face Hub."""

from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo


def upload_to_huggingface(
    file_path: str | Path,
    repo_id: str,
    path_in_repo: Optional[str] = None,
    repo_type: str = "dataset",
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
    create_repo_if_missing: bool = True,
    private: bool = False,
) -> str:
    """
    Upload a file to Hugging Face Hub.

    Args:
        file_path: Local path to the file to upload
        repo_id: Repository ID on Hugging Face (e.g., "username/repo-name")
        path_in_repo: Path where the file should be stored in the repo.
                     If None, uses the filename from file_path
        repo_type: Type of repository ("dataset", "model", or "space")
        token: Hugging Face API token. If None, uses token from HF_TOKEN env var
               or cached credentials
        commit_message: Custom commit message. If None, auto-generates one
        create_repo_if_missing: If True, creates the repository if it doesn't exist
        private: If creating a new repo, whether it should be private

    Returns:
        URL of the uploaded file

    Example:
        >>> upload_to_huggingface(
        ...     file_path="my_recording.h5",
        ...     repo_id="myusername/elden-ring-gameplay",
        ...     repo_type="dataset",
        ...     token="hf_xxx"
        ... )
        'https://huggingface.co/datasets/myusername/elden-ring-gameplay/blob/main/my_recording.h5'
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Initialize Hugging Face API
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist and flag is set
    if create_repo_if_missing:
        try:
            create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                private=private,
                token=token,
                exist_ok=True,
            )
        except Exception as e:
            print(f"Note: Could not create/verify repository: {e}")
    
    # Use filename if path_in_repo not specified
    if path_in_repo is None:
        path_in_repo = file_path.name
    
    # Generate commit message if not provided
    if commit_message is None:
        commit_message = f"Upload {file_path.name}"
    
    # Upload the file
    url = api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
        token=token,
    )
    
    print(f"✓ Successfully uploaded {file_path.name} to {repo_id}")
    print(f"  URL: {url}")
    
    return url


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: uv run -m gameplay_pipeline.hf_upload <file_path> <repo_id> [token]")
        print("Example: uv run -m gameplay_pipeline.hf_upload my_recording.h5 username/repo-name")
        sys.exit(1)
    
    file_path = sys.argv[1]
    repo_id = sys.argv[2]
    token = sys.argv[3] if len(sys.argv) > 3 else None
    
    upload_to_huggingface(
        file_path=file_path,
        repo_id=repo_id,
        token=token,
    )

