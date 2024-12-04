import os
import shutil
from datetime import datetime

def clean_folders(directory_path, keep_num=5):
    """
    保留最新的N个文件夹，删除其他的
    
    Args:
        directory_path: 要处理的目录路径
        keep_num: 要保留的文件夹数量（默认为5）
    """
    try:
        # 获取所有文件夹及其修改时间
        folders = []
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isdir(item_path):  # 只处理文件夹
                mtime = os.path.getmtime(item_path)
                folders.append((item_path, mtime))

        # 按修改时间排序（从新到旧）
        folders.sort(key=lambda x: x[1], reverse=True)

        # 如果文件夹数量不足keep_num，直接返回
        if len(folders) <= keep_num:
            print(f"\n文件夹数量（{len(folders)}）不足 {keep_num}，全部保留")
            return

        # 删除多余的文件夹
        folders_to_delete = folders[keep_num:]
        for folder_path, mtime in folders_to_delete:
            print(f"删除: {folder_path}")
            shutil.rmtree(folder_path)

    except Exception as e:
        print(f"发生错误: {str(e)}")

