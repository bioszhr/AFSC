""" pca_lowrank """
import math
import numpy as np
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
import torch
from sklearn.neighbors import NearestNeighbors 
import torch_geometric
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def large_mat_mul(input_a, input_b, batch=32):
    """
    Large Matrix Slicing Operations.
    """
    m = input_a.shape[0]
    block_m = math.floor(m / batch)
    out = []
    for i in range(batch):
        start = i * block_m
        end = (i + 1) * block_m
        new_a = input_a[start:end]
        out_i = np.matmul(new_a, input_b)
        out.append(out_i)
    out = np.concatenate(out, axis=0)
    remain_a = input_a[batch * block_m:m]
    remain_o = np.matmul(remain_a, input_b)
    output = np.concatenate((out, remain_o), axis=0)
    return output


def mat_mul(input_a, input_b):
    """
    Refactored matmul operation.
    """
    m = input_a.shape[0]
    if m > 100000:
        out = large_mat_mul(input_a, input_b)
    else:
        out = np.matmul(input_a, input_b)

    return out


def get_approximate_basis(matrix: np.ndarray,
                          q=6,
                          niter=2,
                          ):
    """
       Return tensor Q with k orthonormal columns \
       such that 'Q Q^H matrix` approximates `matrix`.
    """
    niter = 2 if niter is None else niter
    _, n = matrix.shape[-2:]

    r = np.random.randn(n, q)

    matrix_t = matrix.T

    q, _ = np.linalg.qr(mat_mul(matrix, r))
    for _ in range(niter):
        q = np.linalg.qr(mat_mul(matrix_t, q))[0]
        q = np.linalg.qr(mat_mul(matrix, q))[0]
    return q


def pca(matrix: np.ndarray, k: int = None, niter: int = 2, norm: bool = False):
    r"""
    Perform a linear principal component analysis (PCA) on the matrix,
    and will return the first k dimensionality-reduced features.

    Args:
      matrix(ndarray): Input features, shape:(B, F)
      k(int): target dimension for dimensionality reduction
      niter(int): the number of subspace iterations to conduct \
      and it must be a nonnegative integer.
      norm(bool): Whether the output is normalized

    Return:
        Tensor, Features after dimensionality reduction

    Example:
        >>> import numpy as np
        >>> X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> data = pca(X, 1)
        >>> print(data)
            [[ 0.33702252]
            [ 2.22871406]
            [ 3.6021826 ]
            [-1.37346854]
            [-2.22871406]
            [-3.6021826 ]]
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("The matrix type is {},\
                        but it should be ndarray.".format(type(matrix)))
    if not isinstance(k, int):
        raise TypeError("The k type is {},\
                        but it should be int.".format(type(k)))
    m, n = matrix.shape[-2:]
    if k is None:
        k = min(6, m, n)

    c = np.mean(matrix, axis=-2)
    norm_matrix = matrix - c

    q = get_approximate_basis(norm_matrix.T, k, niter)
    q_c = q.conjugate()
    b_t = mat_mul(norm_matrix, q_c)
    _, _, v = np.linalg.svd(b_t, full_matrices=False)
    v_c = v.conj().transpose(-2, -1)
    v_c = mat_mul(q, v_c)

    if not norm:
        matrix = mat_mul(matrix, v_c)
    else:
        matrix = mat_mul(norm_matrix, v_c)

    return matrix

def read_adata(file_fold,file_name):
    adata = sc.read_visium(file_fold, count_file=file_name, load_images=True)
    adata.var_names_make_unique()
    adata.X = adata.X.toarray()
    print(adata)
    return adata

def read_label(adata,file_path,lable_name):
    pd_label = pd.read_csv(file_path,sep='\t')
    df_label = pd.DataFrame(pd_label,columns=[lable_name])
    label = pd.Categorical(df_label[lable_name]).codes
    adata.obsm['label']=label
    print(adata)
    return adata


def process_adata(adata,pca_dim=1000,k=50):
    #标准化
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=10000.0)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.X = (adata.X - adata.X.mean(0)) / (adata.X.std(0) + 1e-15)
    gene_tensor = pca(adata.X, pca_dim)
    adata.obsm["X_pca"] = gene_tensor
    #位置坐标
    position = np.ascontiguousarray(adata.obsm["spatial"]) 
    #计算节点距离
    DIS = squareform(pdist(position))
    adata.obsm["distance"] = DIS
    #k个邻居
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(position)  
    _ , indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(k)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    #transform adj to symmetrical adj
    adj_k = interaction
    adj_k = adj_k + adj_k.T
    adj_k = np.where(adj_k>1, 1, adj_k)
    # print(adj,adj.shape,np.count_nonzero(adj))
    adata.obsm['adj_k'] = adj_k
    #计算knn节点距离
    DIS_K=np.multiply(DIS,adj_k)
    adata.obsm['distance_k'] = DIS_K
    print(adata)
    return adata

def prepare_data(adata,need_k,threshold):
    #是否k邻居
    if need_k == 0:
        adj=adata.obsm["distance"]
    else:
        adj=adata.obsm['distance_k']
    adj[adj > threshold]=0
    adj_dis = adj.ravel()[np.flatnonzero(adj)]

    adj_dis = 1000-adj_dis
    
    edge_index = sp.coo_matrix(adj)
    values = edge_index.data 
    indices = np.vstack((edge_index.row, edge_index.col)) # 我们真正需要的coo形式 
    edge_index = torch.LongTensor(indices) # PyG框架需要的

    edge_attr = adj_dis        #不归一化
    edge_attr = torch.tensor(edge_attr).float()
    data = torch_geometric.data.Data(edge_index=edge_index, edge_attr=edge_attr, x=adata.obsm["X_pca"], y = adata.obsm["label"],
                     neighbor_index=edge_index, neighbor_attr=edge_attr)
    return data

def refine(adata,sample_id, pred, shape="hexagon"):
    position = np.ascontiguousarray(adata.obsm["spatial"]) 
    dis = squareform(pdist(position))
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6 
    elif shape=="square":
        num_nbs=4
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:           
            refined_pred.append(self_pred)
    return refined_pred