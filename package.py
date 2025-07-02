import os
import shutil
import zipfile
from datetime import datetime

def create_package():
    # 创建打包时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"news_classification_{timestamp}"
    
    # 创建临时目录
    if not os.path.exists("dist"):
        os.makedirs("dist")
    
    # 创建zip文件
    zip_path = os.path.join("dist", f"{package_name}.zip")
    
    # 需要包含的文件和目录
    include_paths = [
        ("api", "api"),
        ("data/models", "data/models"),
        ("src", "src"),
        ("requirements.txt", "requirements.txt"),
        ("README.md", "README.md")
    ]
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 添加文件到zip
        for src, dest in include_paths:
            if os.path.isfile(src):
                zipf.write(src, dest)
            elif os.path.isdir(src):
                for root, _, files in os.walk(src):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join(dest, os.path.relpath(file_path, src))
                        zipf.write(file_path, arcname)
    
    print(f"Package created successfully: {zip_path}")
    return zip_path

if __name__ == "__main__":
    create_package() 