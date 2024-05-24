import os

def delete_pdb_files_from_subdirs(parent_dir):
    # 遍历parent_dir下的所有文件和文件夹
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            # 检查文件扩展名是否为.pdb
            if file.endswith('.pdb'):
                # 构造完整的文件路径
                file_path = os.path.join(root, file)
                # 删除文件
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except OSError as e:
                    print(f"Error: {file_path} : {e.strerror}")
    print('deleted')
# 指定你的data文件夹路径
data_path = 'PDBind_without_pdb'
delete_pdb_files_from_subdirs(data_path)