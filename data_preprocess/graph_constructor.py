import os  
import glob
import numpy as np  
from tqdm import tqdm 
from openbabel import pybel  
from scipy.spatial import distance  
import torch  
from torch_geometric.nn import radius  
from torch_geometric.utils import remove_self_loops  
from utils.featurizer import Featurizer 

def find_interacting_atoms(decoy, target, cutoff=6.0):
    distances = distance.cdist(decoy, target)  # 计算两个数组中点之间的距离
    decoy_atoms, target_atoms = np.nonzero(distances < cutoff)  # 找出距离小于截断值的点的索引
    decoy_atoms, target_atoms = decoy_atoms.tolist(), target_atoms.tolist()  # 将索引转换为列表
    return decoy_atoms, target_atoms  # 返回相互作用的原子索引

def pocket_atom_counter(path,name):
    n=0
    with open('%s/%s_pocket.mol2'% (path, name)) as f:
        for line in f:
            if '<TRIPOS>ATOM' in line:
                break
        for line in f:
            counter=line.split()
            if '<TRIPOS>BOND' in line or counter[7] == 'HOH':
                break
            n = n+int(counter[5][0] !='H')
    return n
 
def graph_constructor_dir(data_path,save_path,cutoff=6.0):
    pybel.ob.obErrorLog.StopLogging()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for dir_name in os.listdir(data_path):
        subdir_path = os.path.join(data_path, dir_name)
        if os.path.isdir(subdir_path):
            '''使用Featurizer提取分子特征'''
            featurizer=Featurizer(save_molecule_codes=False)  # 创建Featurizer实例，用于特征提取
            charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')
            
            read_ligand=os.path.join(subdir_path, dir_name + '_ligand.mol2')
            ligand = next(pybel.readfile('mol2',read_ligand ))
            ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
            
            read_pocket=os.path.join(subdir_path, dir_name + '_pocket.mol2')
            pocket = next(pybel.readfile('mol2',read_pocket ))
            pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
            if pocket_features.shape[0] !=0:
                if pocket_coords.shape[0] !=0:
                    '''获取口袋中原子的数量，并根据这个数量截取坐标和特征'''
                    node_num = pocket_atom_counter(path=subdir_path, name=dir_name)
                    pocket_coords = pocket_coords[:node_num]
                    pocket_features = pocket_features[:node_num]
                    
                    '''确保配体和口袋的特征和坐标数量匹配'''
                    try:
                        assert (ligand_features[:, charge_idx] != 0).any()
                    except AssertionError:
                        print("Assertion failed: All partial charges are zero.")

                    try:
                        assert (pocket_features[:, charge_idx] != 0).any()
                    except AssertionError:
                        print("Assertion failed: All  are zero.")
                                    
                    assert (ligand_features[:, :9].sum(1) != 0).all()
                    assert ligand_features.shape[0] == ligand_coords.shape[0]
                    assert pocket_features.shape[0] == pocket_coords.shape[0]
                    '''找出相互作用的口袋和配体原子和相互作用原子'''
                    pocket_interact, ligand_interact = find_interacting_atoms(pocket_coords, ligand_coords, cutoff)
                    pocket_atoms = set([])
                    pocket_atoms = pocket_atoms.union(set(pocket_interact))
                    ligand_atoms = range(len(ligand_coords))  # 创建配体原子的索引集合

                    '''将口袋原子的集合转换为numpy数组,并根据这个数组截取口袋的坐标和特征,构建torch张量'''
                    pocket_atoms = np.array(list(pocket_atoms))
                    pocket_atoms = pocket_atoms.astype(int)
                    pocket_coords = pocket_coords[pocket_atoms]
                    pocket_features = pocket_features[pocket_atoms]
                    ligand_pos = np.array(ligand_coords)
                    pocket_pos = np.array(pocket_coords)
                    '''将口袋的位置信息转换为torch张量,并使用半径图卷积构建边'''
                    pos = torch.tensor(pocket_pos)
                    row, col = radius(pos, pos, 0.5, max_num_neighbors=1000)
                    full_edge_index_long = torch.stack([row, col], dim=0)  # 获取边的索引
                    full_edge_index_long, _ = remove_self_loops(full_edge_index_long)  # 移除自环
                    if full_edge_index_long.size()[1] > 0:  # 如果有边
                        j_long, i_long = full_edge_index_long  # 分别获取边的索引
                        pocket_pos = np.delete(pocket_pos, j_long[:len(j_long)//2], axis=0)  # 删除一半的边
                        pocket_features = np.delete(pocket_features, j_long[:len(j_long)//2], axis=0)  # 删除对应的特征

                    '''串联三个子图的位置和特征'''
                    complex_pos = np.concatenate((pocket_pos, ligand_pos), axis=0)
                    complex_features = np.concatenate((pocket_features, ligand_features), axis=0)

                    x_shift = np.mean(complex_pos[:, 0])
                    complex_pos -= [x_shift, 0.0, 0.0]
                    pocket_pos -= [x_shift, 0.0, 0.0]
                    ligand_pos -= [x_shift, 0.0, 0.0]

                    
                    pocket_pos += [100.0, 0.0, 0.0]    # 口袋子图沿x轴移动100埃
                    ligand_pos += [200.0, 0.0, 0.0]    # 配体子图沿x轴移动200埃

                    
                    final_pos = np.concatenate((complex_pos, pocket_pos, ligand_pos), axis=0)
                    final_features = np.concatenate((complex_features, pocket_features, ligand_features), axis=0)

                    # 生成用于加载图的文件

                    indicator = np.array([[dir_name] for _ in range(final_features.shape[0])])
                    # 保存指示器、节点标签、节点属性和图标签到文件
                    os.makedirs(os.path.join(save_path, dir_name), exist_ok=True)
                    with open(os.path.join(save_path, dir_name,f"{dir_name}_graph_indicator.txt"),'ab') as f:
                        np.savetxt(f, indicator, fmt='%s', delimiter=', ')
                    f.close()

                    with open(os.path.join(save_path, dir_name,f"{dir_name}_node_labels.txt"),'ab') as f:
                        np.savetxt(f, final_features, fmt='%.4f', delimiter=', ')
                    f.close()

                    with open(os.path.join(save_path, dir_name,f"{dir_name}_node_attributes.txt"),'ab') as f:
                        np.savetxt(f, final_pos, fmt='%.3f', delimiter=', ')
            
    print('finished')
def main():
    graph_constructor_dir(data_path='PDBind_without_pdb',save_path='PDBgraph',cutoff=6.0)
if __name__ == "__main__":
    main()