from huggingface_hub import HfApi

# 1. 初始化 API 
api = HfApi()

# 2. 设置你的仓库信息
# 格式为 "用户名/仓库名"，例如 "your_name/algebra2005"
repo_id = "Lucy9999/AL2005" 
local_folder_path = r"E:\dkt1111\Q-MCKT\data\algebra2005"

# 3. 创建仓库（如果还没创建的话，已创建则会跳过）
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

# 4. 上传文件夹
print(f"正在上传 {local_folder_path} 到 {repo_id}...")
api.upload_folder(
    folder_path=local_folder_path,
    repo_id=repo_id,
    repo_type="dataset",
)
print("上传完成！")