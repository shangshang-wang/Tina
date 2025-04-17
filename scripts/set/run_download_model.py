from huggingface_hub import snapshot_download
import os


if __name__ == "__main__":
    CKPT_DIR = os.environ['CKPT_DIR']

    print("Downloading deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B ...")
    snapshot_download(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                      local_dir=f"{CKPT_DIR}/models/DeepSeek-R1-Distill-Qwen-1.5B/base")
