from huggingface_hub import HfApi, Repository, create_repo, upload_file

class HuggingfaceUtils:
    def __init__(self, hf_token):
        self.api = HfApi()
        self.hf_token = hf_token

    def create_repository(self, repo_name):
        create_repo(repo_id=repo_name, token=self.hf_token, private=True)

    def upload_model_files(self, repo_name, local_dir):
        repo = Repository(local_dir=local_dir, clone_from=repo_name, use_auth_token=self.hf_token)
        repo.push_to_hub(add_patterns="*", commit_message="Upload fine-tuned model")

    def download_model_files(self, repo_name, local_dir):
        self.api.download_repo(repo_id=repo_name, cache_dir=local_dir, use_auth_token=self.hf_token)
