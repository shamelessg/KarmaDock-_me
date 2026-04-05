#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2022/8/8 14:33
# @author : Xujun Zhang

from copy import deepcopy
import torch
import networkx as nx
import numpy as np
import copy
from torch_geometric.utils import to_networkx
from rdkit import Chem
import warnings
from rdkit.Chem import AllChem
from torch_geometric.utils import to_dense_adj, dense_to_sparse
warnings.filterwarnings("ignore")

def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

def get_higher_order_adj_matrix(adj, order):
    """
    参数:
        adj:        (N, N)
        type_mat:   (N, N)
    返回:
        以下属性将被更新:
            - edge_index
            - edge_type
        以下属性将被添加到数据对象:
            - bond_edge_index:  原始的 edge_index.
    """
    adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

    for i in range(2, order + 1):
        adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
    order_mat = torch.zeros_like(adj)

    for i in range(1, order + 1):
        order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

    return order_mat

def get_ring_adj_matrix(mol, adj):
    new_adj = deepcopy(adj)
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        # print(ring)
        for i in ring:
            for j in ring:
                if i==j:
                    continue
                elif new_adj[i][j] != 1:
                    new_adj[i][j]+=1
    return new_adj




# orderd by perodic table
atom_vocab = ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"]
atom_vocab = {a: i for i, a in enumerate(atom_vocab)}
degree_vocab = range(7)
num_hs_vocab = range(7)
formal_charge_vocab = range(-5, 6)
chiral_tag_vocab = range(4)
total_valence_vocab = range(8)
num_radical_vocab = range(8)
hybridization_vocab = range(len(Chem.rdchem.HybridizationType.values))

bond_type_vocab = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
bond_type_vocab = {b: i for i, b in enumerate(bond_type_vocab)}
bond_dir_vocab = range(len(Chem.rdchem.BondDir.values))
bond_stereo_vocab = range(len(Chem.rdchem.BondStereo.values))

# orderd by molecular mass
residue_vocab = ["GLY", "ALA", "SER", "PRO", "VAL", "THR", "CYS", "ILE", "LEU", "ASN",
                 "ASP", "GLN", "LYS", "GLU", "MET", "HIS", "PHE", "ARG", "TYR", "TRP"]


def onehot(x, vocab, allow_unknown=False):
    if x in vocab:
        if isinstance(vocab, dict):
            index = vocab[x]
        else:
            index = vocab.index(x)
    else:
        index = -1
    if allow_unknown:
        feature = [0] * (len(vocab) + 1)
        if index == -1:
            warnings.warn("Unknown value `%s`" % x)
        feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError("Unknown value `%s`. Available vocabulary is `%s`" % (x, vocab))
        feature[index] = 1

    return feature


def atom_default(atom):
    """默认原子特征。

    特征:
        GetSymbol(): 原子符号的one-hot嵌入 dim=18
        
        GetChiralTag(): 原子手性标签的one-hot嵌入 dim=5
        
        GetTotalDegree(): 分子中原子的度数（包括Hs）的one-hot嵌入 dim=5
        
        GetFormalCharge(): 分子中形式电荷数的one-hot嵌入
        
        GetTotalNumHs(): 原子上Hs的总数（显式和隐式）的one-hot嵌入 
        
        GetNumRadicalElectrons(): 原子上自由基电子数的one-hot嵌入
        
        GetHybridization(): 原子杂化的one-hot嵌入
        
        GetIsAromatic(): 原子是否为芳香性
        
        IsInRing(): 原子是否在环中
        18 + 5 + 8 + 12 + 8 + 9 + 10 + 9 + 3 + 4 
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetChiralTag(), chiral_tag_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetFormalCharge(), formal_charge_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab, allow_unknown=True) + \
           onehot(atom.GetNumRadicalElectrons(), num_radical_vocab, allow_unknown=True) + \
           onehot(atom.GetHybridization(), hybridization_vocab, allow_unknown=True) + \
            onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True) + \
           [atom.GetIsAromatic(), atom.IsInRing(), atom.IsInRing() and (not atom.IsInRingSize(3)) and (not atom.IsInRingSize(4)) \
            and (not atom.IsInRingSize(5)) and (not atom.IsInRingSize(6))]+[atom.IsInRingSize(i) for i in range(3, 7)]


def atom_center_identification(atom):
    """反应中心识别原子特征。

    特征:
        GetSymbol(): 原子符号的one-hot嵌入
        
        GetTotalNumHs(): 原子上Hs的总数（显式和隐式）的one-hot嵌入 
        
        GetTotalDegree(): 分子中原子的度数（包括Hs）的one-hot嵌入
        
        GetTotalValence(): 原子的总价态（显式 + 隐式）的one-hot嵌入
        
        GetIsAromatic(): 原子是否为芳香性
        
        IsInRing(): 原子是否在环中
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab) + \
           onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalValence(), total_valence_vocab) + \
           [atom.GetIsAromatic(), atom.IsInRing()]



def atom_synthon_completion(atom):
    """合成子完成原子特征。

    特征:
        GetSymbol(): 原子符号的one-hot嵌入

        GetTotalNumHs(): 原子上Hs的总数（显式和隐式）的one-hot嵌入 
        
        GetTotalDegree(): 分子中原子的度数（包括Hs）的one-hot嵌入
        
        IsInRing(): 原子是否在环中
        
        IsInRingSize(3, 4, 5, 6): 原子是否在特定大小的环中
        
        IsInRing() and not IsInRingSize(3, 4, 5, 6): 原子是否在环中且不在3, 4, 5, 6大小的环中
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab) + \
           onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True) + \
           [atom.IsInRing(), atom.IsInRingSize(3), atom.IsInRingSize(4),
            atom.IsInRingSize(5), atom.IsInRingSize(6), 
            atom.IsInRing() and (not atom.IsInRingSize(3)) and (not atom.IsInRingSize(4)) \
            and (not atom.IsInRingSize(5)) and (not atom.IsInRingSize(6))]



def atom_symbol(atom):
    """符号原子特征。

    特征:
        GetSymbol(): 原子符号的one-hot嵌入
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True)



def atom_explicit_property_prediction(atom):
    """显式性质预测原子特征。

    特征:
        GetSymbol(): 原子符号的one-hot嵌入

        GetDegree(): 分子中原子的度数的one-hot嵌入

        GetTotalValence(): 原子的总价态（显式 + 隐式）的one-hot嵌入
        
        GetFormalCharge(): 分子中形式电荷数的one-hot嵌入
        
        GetIsAromatic(): 原子是否为芳香性
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True) + \
           onehot(atom.GetFormalCharge(), formal_charge_vocab) + \
           [atom.GetIsAromatic()]



def atom_property_prediction(atom):
    """性质预测原子特征。

    特征:
        GetSymbol(): 原子符号的one-hot嵌入
        
        GetDegree(): 分子中原子的度数的one-hot嵌入
        
        GetTotalNumHs(): 原子上Hs的总数（显式和隐式）的one-hot嵌入 
        
        GetTotalValence(): 原子的总价态（显式 + 隐式）的one-hot嵌入
        
        GetFormalCharge(): 分子中形式电荷数的one-hot嵌入
        
        GetIsAromatic(): 原子是否为芳香性
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True) + \
           onehot(atom.GetFormalCharge(), formal_charge_vocab, allow_unknown=True) + \
           [atom.GetIsAromatic()]



def atom_position(atom):
    """
    分子构象中的原子位置。
    如果可用，返回3D位置，否则返回2D位置。

    注意，计算大分子的构象需要很多时间。
    """
    mol = atom.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    pos = conformer.GetAtomPosition(atom.GetIdx())
    return [pos.x, pos.y, pos.z]



def atom_pretrain(atom):
    """预训练的原子特征。

    特征:
        GetSymbol(): 原子符号的one-hot嵌入
        
        GetChiralTag(): 原子手性标签的one-hot嵌入
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetChiralTag(), chiral_tag_vocab)



def atom_residue_symbol(atom):
    """作为原子特征的残基符号。仅支持蛋白质中的原子。

    特征:
        GetSymbol(): 原子符号的one-hot嵌入
        GetResidueName(): 残基符号的one-hot嵌入
    """
    residue = atom.GetPDBResidueInfo()
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(residue.GetResidueName() if residue else -1, residue_vocab, allow_unknown=True)


def bond_default(bond):
    """默认键特征。

    特征:
        GetBondType(): 键类型的one-hot嵌入
        
        GetBondDir(): 键方向的one-hot嵌入
        
        GetStereo(): 键的立体构型的one-hot嵌入
        
        GetIsConjugated(): 键是否被认为是共轭的
    """
    return onehot(bond.GetBondType(), bond_type_vocab, allow_unknown=True) + \
           onehot(bond.GetBondDir(), bond_dir_vocab) + \
           onehot(bond.GetStereo(), bond_stereo_vocab, allow_unknown=True) + \
           [int(bond.GetIsConjugated())]



def bond_length(bond):
    """
    分子构象中的键长。

    注意，计算大分子的构象需要很多时间。
    """
    mol = bond.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    h = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
    t = conformer.GetAtomPosition(bond.GetEndAtomIdx())
    return [h.Distance(t)]



def bond_property_prediction(bond):
    """性质预测键特征。

    特征:
        GetBondType(): 键类型的one-hot嵌入
        
        GetIsConjugated(): 键是否被认为是共轭的
        
        IsInRing(): 键是否在环中
    """
    return onehot(bond.GetBondType(), bond_type_vocab) + \
           [int(bond.GetIsConjugated()), bond.IsInRing()]



def bond_pretrain(bond):
    """预训练的键特征。

    特征:
        GetBondType(): 键类型的one-hot嵌入
        
        GetBondDir(): 键方向的one-hot嵌入
    """
    return onehot(bond.GetBondType(), bond_type_vocab) + \
           onehot(bond.GetBondDir(), bond_dir_vocab)



def residue_symbol(residue):
    """符号残基特征。

    特征:
        GetResidueName(): 残基符号的one-hot嵌入
    """
    return onehot(residue.GetResidueName(), residue_vocab, allow_unknown=True)



def residue_default(residue):
    """默认残基特征。

    特征:
        GetResidueName(): 残基符号的one-hot嵌入
    """
    return residue_symbol(residue)



def ExtendedConnectivityFingerprint(mol, radius=2, length=1024):
    """扩展连接指纹分子特征。

    特征:
        GetMorganFingerprintAsBitVect(): 分子的Morgan指纹作为位向量
    """
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, length)
    return list(ecfp)




def molecule_default(mol):
    """默认分子特征。"""
    return ExtendedConnectivityFingerprint(mol)


ECFP = ExtendedConnectivityFingerprint

def get_full_connected_edge(frag):
    frag = np.asarray(list(frag))
    return torch.from_numpy(np.repeat(frag, len(frag)-1)), \
        torch.from_numpy(np.concatenate([np.delete(frag, i) for i in range(frag.shape[0])], axis=0))

def remove_repeat_edges(new_edge_index, refer_edge_index, N_atoms):
    new = to_dense_adj(new_edge_index, max_num_nodes=N_atoms)
    ref = to_dense_adj(refer_edge_index, max_num_nodes=N_atoms)
    delta_ = new - ref
    delta_[delta_<1] = 0
    unique, _ = dense_to_sparse(delta_)
    return unique

def get_ligand_feature(mol):
    xyz = mol.GetConformer().GetPositions()
    # covalent
    node_feature = []
    edge_index = []
    edge_feature = []
    for idx, atom in enumerate(mol.GetAtoms()):
        # node
        node_feature.append(atom_default(atom))
        # edge
        for bond in atom.GetBonds():
            edge_feature.append(bond_default(bond))
            for bond_idx in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
                # if bond_idx != idx and bond_idx > idx:  # 单向
                if bond_idx != idx:  # 双向
                    edge_index.append([idx, bond_idx])
    node_feature = torch.from_numpy(np.asanyarray(node_feature)).float()  # nodes_chemical_features
    edge_index = torch.from_numpy(np.asanyarray(edge_index).T).long()
    edge_feature = torch.from_numpy(np.asanyarray(edge_feature)).float()
    # 0:4 bond type 5:11 bond direction 12:18 bond stero 19 bond conjunction
    l_full_edge_s = edge_feature[:, :5]
    xyz = torch.from_numpy(xyz)
    l_full_edge_s = torch.cat([l_full_edge_s, torch.pairwise_distance(xyz[edge_index[0]], xyz[edge_index[1]]).unsqueeze(dim=-1)], dim=-1)
    # ring
    adj = to_dense_adj(edge_index).squeeze(0)
    adj_ring = get_ring_adj_matrix(mol, adj)
    adj_ring = adj_ring - adj
    if adj_ring.any():
        (u_ring, v_ring), _ = dense_to_sparse(adj_ring)
        edge_index_new = torch.stack([u_ring, v_ring], dim=0)
        edge_index = torch.cat([edge_index, edge_index_new], dim=1)
        edge_feature_new = torch.zeros((edge_index_new.size(1), 20))
        edge_feature_new[:, [4, 5, 18]] = 1
        edge_feature = torch.cat([edge_feature, edge_feature_new], dim=0)
        l_full_edge_s_new = torch.cat([edge_feature_new[:, :5], torch.pairwise_distance(xyz[edge_index_new[0]], xyz[edge_index_new[1]]).unsqueeze(dim=-1)], dim=-1)
        l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    # interaction
    adj_interaction = get_higher_order_adj_matrix(adj, order=3)
    adj_interaction = adj_interaction - adj
    (u_interaction, v_interaction), _ = dense_to_sparse(adj_interaction)
    edge_index_new = torch.stack([u_interaction, v_interaction], dim=0)
    edge_index = torch.cat([edge_index, edge_index_new], dim=1)
    edge_feature_new = torch.zeros((edge_index_new.size(1), 20))
    edge_feature_new[:, [4, 5, 18]] = 1
    edge_feature = torch.cat([edge_feature, edge_feature_new], dim=0)
    interaction_edge_mask = torch.ones((edge_feature.size(0),))
    interaction_edge_mask[-edge_feature_new.size(0):] = 0
    l_full_edge_s_new = torch.cat([edge_feature_new[:, :5], -torch.ones(edge_feature_new.size(0), 1)], dim=-1)
    l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    x = (xyz, node_feature, edge_index, edge_feature, l_full_edge_s, interaction_edge_mask.bool()) 
    return x 

def get_ligand_feature_v1(mol, use_chirality=True):
    xyz = mol.GetConformer().GetPositions()
    # covalent
    N_atoms = mol.GetNumAtoms()
    node_feature = []
    edge_index = []
    edge_feature = []
    G = nx.Graph()
    for idx, atom in enumerate(mol.GetAtoms()):
        # node
        node_feature.append(atom_default(atom))
        # edge
        for bond in atom.GetBonds():
            edge_feature.append(bond_default(bond))
            for bond_idx in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
                # if bond_idx != idx and bond_idx > idx:  # 单向
                if bond_idx != idx:  # 双向
                    edge_index.append([idx, bond_idx])
                    G.add_edge(idx, bond_idx)
    node_feature = torch.from_numpy(np.asanyarray(node_feature)).float()  # nodes_chemical_features
    if use_chirality:
        try:
            chiralcenters = Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=True, useLegacyImplementation=False)
        except:
            chiralcenters = []
        chiral_arr = torch.zeros([N_atoms,3]) 
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] =1 
            elif rs == 'S':
                chiral_arr[i, 1] =1 
            else:
                chiral_arr[i, 2] =1 
        node_feature = torch.cat([node_feature,chiral_arr.float()],dim=1)
    edge_index = torch.from_numpy(np.asanyarray(edge_index).T).long()
    edge_feature = torch.from_numpy(np.asanyarray(edge_feature)).float()
    cov_edge_num = edge_index.size(1)
    # 0:4 bond type 5:11 bond direction 12:18 bond stero 19 bond conjunction
    l_full_edge_s = edge_feature[:, :5]
    xyz = torch.from_numpy(xyz)
    l_full_edge_s = torch.cat([l_full_edge_s, torch.pairwise_distance(xyz[edge_index[0]], xyz[edge_index[1]]).unsqueeze(dim=-1)], dim=-1)
    # get fragments based on rotation bonds
    frags = []
    rotate_bonds = []
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2): continue
        # print(f'{sorted(nx.connected_components(G2), key=len)[0]}|{sorted(nx.connected_components(G2), key=len)[1]}')
        l = (sorted(nx.connected_components(G2), key=len)[0])
        if len(l) < 2: continue
        else:
            # rotate_bonds.append(e)
            # rotate_bonds.append((e[1], e[0]))
            frags.append(l)
    if len(frags) != 0:
        frags = sorted(frags, key=len)
        for i in range(len(frags)):
            for j in range(i+1, len(frags)):
                frags[j] -= frags[i]
        frags = [i for i in frags if len(i) > 1]
        frag_edge_index = torch.cat([torch.stack(get_full_connected_edge(i), dim=0) for i in frags], dim=1).long()
        edge_index_new = remove_repeat_edges(new_edge_index=frag_edge_index, refer_edge_index=edge_index, N_atoms=N_atoms)
        edge_index = torch.cat([edge_index, edge_index_new], dim=1)
        edge_feature_new = torch.zeros((edge_index_new.size(1), 20))
        edge_feature_new[:, [4, 5, 18]] = 1
        edge_feature = torch.cat([edge_feature, edge_feature_new], dim=0)
        l_full_edge_s_new = torch.cat([edge_feature_new[:, :5], torch.pairwise_distance(xyz[edge_index_new[0]], xyz[edge_index_new[1]]).unsqueeze(dim=-1)], dim=-1)
        l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    # interaction
    adj_interaction = torch.ones((N_atoms, N_atoms)) - torch.eye(N_atoms, N_atoms)
    interaction_edge_index, _ = dense_to_sparse(adj_interaction)
    edge_index_new = remove_repeat_edges(new_edge_index=interaction_edge_index, refer_edge_index=edge_index, N_atoms=N_atoms)
    edge_index = torch.cat([edge_index, edge_index_new], dim=1)
    edge_feature_new = torch.zeros((edge_index_new.size(1), 20))
    edge_feature_new[:, [4, 5, 18]] = 1
    edge_feature = torch.cat([edge_feature, edge_feature_new], dim=0)
    interaction_edge_mask = torch.ones((edge_feature.size(0),))
    interaction_edge_mask[-edge_feature_new.size(0):] = 0
    # scale the distance
    l_full_edge_s[:, -1] *= 0.1
    l_full_edge_s_new = torch.cat([edge_feature_new[:, :5], -torch.ones(edge_feature_new.size(0), 1)], dim=-1)
    l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    # cov edge mask
    cov_edge_mask = torch.zeros(edge_feature.size(0),)
    cov_edge_mask[:cov_edge_num] = 1
    x = (xyz, node_feature, edge_index, edge_feature, l_full_edge_s, interaction_edge_mask.bool(), cov_edge_mask.bool()) 
    return x 
