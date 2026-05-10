from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Lucy9999/junyi2015",
    repo_type="dataset",
    local_dir=".",
    # 关键设置：减少并发，降低内存压力
    max_workers=1, 
    resume_download=True
)