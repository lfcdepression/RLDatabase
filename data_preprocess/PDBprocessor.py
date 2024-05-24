from Bio.PDB import Select,NeighborSearch,PDBIO,PDBParser
import os
import numpy as np
from openbabel import pybel

io=PDBIO()
def is_ligand(residue):
    non_ligand_residues = {'HOH', 'DOD', 'HEM', 'ACP', 'FAD', 'NAD', 'SAM', 'SO3',
                        'PO4', 'COA', 'PLP', 'GTP', 'ADP', 'PEP', 'ACT', 'PEG', 'ION'}

    return residue.id[0] != ' ' and residue.resname not in non_ligand_residues

def is_rna_residue(residue):
    rna_residues = {'A', 'ADE', 'U', 'URA', 'G', 'GUA', 'C', 'CYT'}
    return residue.id[0] == ' ' and residue.resname in rna_residues

def get_ligand_center(structure):
    ligand_atoms = [atom for atom in structure.get_atoms() if is_ligand(atom.get_parent())]
    ligand_coords = np.array([atom.coord for atom in ligand_atoms])
    return np.mean(ligand_coords, axis=0)

class RNA_Residue_Selector(Select):
    def __init__(self,residue) :
        self.residue = residue
    def accept_residue(self, residue):
        return  is_rna_residue(residue)

class Ligand_Residue_Selector(Select):
    def __init__(self, residue):
        self.residue = residue
    def accept_residue(self, residue):
        return is_ligand(residue)
 
class Pocket_Residue_Selector(Select):
    def __init__(self,structure,cutoff=10.0):
        self.structure = structure
        self.cutoff=cutoff
        self.pocket_residues = self.pocket_finder()
    def pocket_finder(self):
        ligand_center=get_ligand_center(self.structure)
        kd = NeighborSearch(list(self.structure.get_atoms()))
        rna_pocket_residues = kd.search(ligand_center, self.cutoff, level="R") 
        return [residue for residue in rna_pocket_residues ]
    def accept_residue(self, residue):
        return residue in self.pocket_residues and is_rna_residue(residue)

def save_rna(structure,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    io.set_structure(structure)
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_ligand(residue):
                    selector=RNA_Residue_Selector(residue)
                    filename = f'{structure.id}_modeled.pdb'
                    full_path = os.path.join(save_path, filename)
                    io.save(full_path, selector)
    print(f"RNA residues saved as {full_path}")

def save_ligand(structure,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    io.set_structure(structure)
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_ligand(residue):
                    selector=Ligand_Residue_Selector(residue)
                    filename = f'{structure.id}_ligand.pdb'
                    full_path = os.path.join(save_path,  filename)
                    io.save(full_path, selector)
                    # 将PDB文件转换为MOL2文件
                    pdb = next(pybel.readfile("pdb", full_path))
                    mol2_filename = f'{structure.id}_ligand.mol2'
                    mol2_full_path = os.path.join(save_path, mol2_filename)
                    output_mol2 = pybel.Outputfile("mol2", mol2_full_path, overwrite=True)
                    output_mol2.write(pdb)  # 假设只有一个分子
                    output_mol2.close()
    print(f"Ligand residues saved as {full_path}")                
                    

    
def save_pocket(structure,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    selector=Pocket_Residue_Selector(structure,cutoff=10.0)
    io.set_structure(structure)
    filename = f'{structure.id}_pocket.pdb'
    full_path = os.path.join(save_path, filename)
    io.save(full_path, selector)
    # 将PDB文件转换为MOL2文件
    pdb = next(pybel.readfile("pdb", full_path))
    mol2_filename = f'{structure.id}_pocket.mol2'
    mol2_full_path = os.path.join(save_path, mol2_filename)
    output_mol2 = pybel.Outputfile("mol2", mol2_full_path, overwrite=True)
    output_mol2.write(pdb)  # 假设只有一个分子
    output_mol2.close()
    print(f"Pocket residues saved as {full_path}")
    

def main():
    
    parser = PDBParser()
    pdbind_path = 'PDBind'  # 目标文件夹
    data_path = 'data'      # 源文件夹，包含PDB文件
    for filename in os.listdir(data_path):
        if filename.endswith('.pdb'):  # 确保处理的是PDB文件
            pdb_id = os.path.splitext(filename)[0]  # 获取PDB文件的ID（无扩展名）
            pdb_file_path = os.path.join(data_path, filename) 
            structure = parser.get_structure(pdb_id, pdb_file_path)
            save_path = os.path.join(pdbind_path, pdb_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_ligand(structure, save_path)
            save_rna(structure, save_path)
            save_pocket(structure, save_path)

if __name__ == "__main__":
    main()