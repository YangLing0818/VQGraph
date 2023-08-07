"""
Dataloader of CPF datasets are adapted from the CPF implementation
https://github.com/BUPT-GAMMA/CPF/tree/389c01aaf238689ee7b1e5aba127842341e123b6/data

Dataloader of NonHom datasets are adapted from the Non-homophily benchmarks
https://github.com/CUAI/Non-Homophily-Benchmarks

Dataloader of BGNN datasets are adapted from the BGNN implementation and dgl example of BGNN
https://github.com/nd7141/bgnn
https://github.com/dmlc/dgl/tree/473d5e0a4c4e4735f1c9dc9d783e0374328cca9a/examples/pytorch/bgnn
"""

import numpy as np
import scipy.sparse as sp
import torch
import dgl
import os
import scipy
import pandas as pd
import json
from dgl.data.utils import load_graphs
from os import path
from category_encoders import CatBoostEncoder
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing
from data_preprocess import (
    normalize_adj,
    eliminate_self_loops_adj,
    largest_connected_components,
    binarize_labels,
)
from ogb.nodeproppred import DglNodePropPredDataset


CPF_data = ["cora", "citeseer", "pubmed", "a-computer", "a-photo"]
OGB_data = ["ogbn-arxiv", "ogbn-products"]
NonHom_data = ["pokec", "penn94"]
BGNN_data = ["house_class", "vk_class"]


def load_data(dataset, dataset_path, **kwargs):
    if dataset in CPF_data:
        return load_cpf_data(
            dataset,
            dataset_path,
            kwargs["seed"],
            kwargs["labelrate_train"],
            kwargs["labelrate_val"],
        )
    elif dataset in OGB_data:
        return load_ogb_data(dataset, dataset_path)
    elif dataset in NonHom_data:
        return load_nonhom_data(dataset, dataset_path, kwargs["split_idx"])
    elif dataset in BGNN_data:
        return load_bgnn_data(dataset, dataset_path, kwargs["split_idx"])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_ogb_data(dataset, dataset_path):
    data = DglNodePropPredDataset(dataset, dataset_path)
    splitted_idx = data.get_idx_split()
    idx_train, idx_val, idx_test = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )

    g, labels = data[0]
    labels = labels.squeeze()

    # Turn the graph to undirected
    if dataset == "ogbn-arxiv":
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)
        g = g.remove_self_loop().add_self_loop()
        
    if dataset == "ogbn-products":
        g = g.add_self_loop()

    return g, labels, idx_train, idx_val, idx_test


def load_cpf_data(dataset, dataset_path, seed, labelrate_train, labelrate_val):
    data_path = Path.cwd().joinpath(dataset_path, f"{dataset}.npz")
    if os.path.isfile(data_path):
        data = load_npz_to_sparse_graph(data_path)
    else:
        raise ValueError(f"{data_path} doesn't exist.")

    # remove self loop and extract the largest CC
    data = data.standardize()
    adj, features, labels = data.unpack()
    # print(labels[696], labels[701])
    # print(features.dtype)
    # print(adj[165])
    # print(adj[1989])
    
    # print(adj[516])
    # print(adj[1829])
    labels = binarize_labels(labels)
    # print(])
    # print(features[696]==features[701])
    random_state = np.random.RandomState(seed)
    idx_train, idx_val, idx_test = get_train_val_test_split(
        random_state, labels, labelrate_train, labelrate_val
    )

    features = torch.FloatTensor(np.array(features.todense()))
    num_nodes = features.shape[0]
    # adj = torch.FloatTensor(np.array(adj.todense()))
    #print(features)
    labels = torch.LongTensor(labels.argmax(axis=1))
    # print(labels[696])
    # for i in range(num_nodes):
    #     for j in range(i+1, num_nodes):
    #         if torch.all(torch.eq(labels[i], labels[j])) and torch.all(torch.eq(adj[i], adj[j]))and not torch.all(torch.eq(features[i], features[j])):
    #                 print(i, j)
    #                 print("true")
    # print("here!")
    # feature_similarity = torch.matmul(features, features.t())
    # # print(feature_similarity.shape)
    # feature_similarity.fill_diagonal_(-1)
    # # get the index of the nodes with the max similarity
    # feature_similarity = feature_similarity.cpu().detach().numpy()
    # max_pair = np.unravel_index(np.argmax(feature_similarity), feature_similarity.shape)
    # print(max_pair)
    adj = normalize_adj(adj)
    adj_sp = adj.tocoo()
    g = dgl.graph((adj_sp.row, adj_sp.col))
    g.ndata["feat"] = features
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    # print(labels)
    return g, labels, idx_train, idx_val, idx_test


def load_nonhom_data(dataset, dataset_path, split_idx):
    data_path = Path.cwd().joinpath(dataset_path, f"{dataset}.mat")
    data_split_path = Path.cwd().joinpath(
        dataset_path, "splits", f"{dataset}-splits.npy"
    )

    if dataset == "pokec":
        g, features, labels = load_pokec_mat(data_path)
    elif dataset == "penn94":
        g, features, labels = load_penn94_mat(data_path)
    else:
        raise ValueError("Invalid dataname")

    g = g.remove_self_loop().add_self_loop()
    g.ndata["feat"] = features
    labels = torch.LongTensor(labels)

    splitted_idx = load_fixed_splits(dataset, data_split_path, split_idx)
    idx_train, idx_val, idx_test = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    return g, labels, idx_train, idx_val, idx_test


def load_bgnn_data(dataset, dataset_path, split_idx):
    data_path = Path.cwd().joinpath(dataset_path, f"{dataset}")

    g, X, y, cat_features, masks = read_input(data_path)
    train_mask, val_mask, test_mask = (
        masks[str(split_idx)]["train"],
        masks[str(split_idx)]["val"],
        masks[str(split_idx)]["test"],
    )

    encoded_X = X.copy()
    if cat_features is not None and len(cat_features):
        encoded_X = encode_cat_features(
            encoded_X, y, cat_features, train_mask, val_mask, test_mask
        )
    encoded_X = normalize_features(encoded_X, train_mask, val_mask, test_mask)
    encoded_X = replace_na(encoded_X, train_mask)
    features, labels = pandas_to_torch(encoded_X, y)

    g = g.remove_self_loop().add_self_loop()
    g.ndata["feat"] = features
    labels = labels.long()

    idx_train = torch.LongTensor(train_mask)
    idx_val = torch.LongTensor(val_mask)
    idx_test = torch.LongTensor(test_mask)
    return g, labels, idx_train, idx_val, idx_test


def load_out_t(out_t_dir, name):
    return torch.from_numpy(np.load(out_t_dir.joinpath(name))["arr_0"])


""" For NonHom"""
dataset_drive_url = {"pokec": "1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y"}
splits_drive_url = {"pokec": "1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_"}


def load_penn94_mat(data_path):
    mat = scipy.io.loadmat(data_path)
    A = mat["A"]
    metadata = mat["local_info"]

    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int)

    # make features into one-hot encodings
    feature_vals = np.hstack((np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    g = dgl.graph((edge_index[0], edge_index[1]))
    g = dgl.to_bidirected(g)

    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(metadata[:, 1] - 1)  # gender label, -1 means unlabeled
    return g, features, labels


def load_pokec_mat(data_path):
    if not path.exists(data_path):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url["pokec"], dest_path=data_path, showsize=True
        )

    fulldata = scipy.io.loadmat(data_path)
    edge_index = torch.tensor(fulldata["edge_index"], dtype=torch.long)
    g = dgl.graph((edge_index[0], edge_index[1]))
    g = dgl.to_bidirected(g)

    features = torch.tensor(fulldata["node_feat"]).float()
    labels = fulldata["label"].flatten()
    return g, features, labels


class NCDataset(object):
    def __init__(self, name, root):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type="random", train_prop=0.5, valid_prop=0.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == "random":
            ignore_negative = False if self.name == "ogbn-proteins" else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label,
                train_prop=train_prop,
                valid_prop=valid_prop,
                ignore_negative=ignore_negative,
            )
            split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, len(self))


def load_fixed_splits(dataset, data_split_path="", split_idx=0):
    if not os.path.exists(data_split_path):
        assert dataset in splits_drive_url.keys()
        gdd.download_file_from_google_drive(
            file_id=splits_drive_url[dataset], dest_path=data_split_path, showsize=True
        )

    splits_lst = np.load(data_split_path, allow_pickle=True)
    splits = splits_lst[split_idx]

    for key in splits:
        if not torch.is_tensor(splits[key]):
            splits[key] = torch.as_tensor(splits[key])

    return splits


"""For BGNN """


def pandas_to_torch(*args):
    return [torch.from_numpy(arg.to_numpy(copy=True)).float().squeeze() for arg in args]


def read_input(input_folder):
    X = pd.read_csv(f"{input_folder}/X.csv")
    y = pd.read_csv(f"{input_folder}/y.csv")

    categorical_columns = []
    if os.path.exists(f"{input_folder}/cat_features.txt"):
        with open(f"{input_folder}/cat_features.txt") as f:
            for line in f:
                if line.strip():
                    categorical_columns.append(line.strip())

    cat_features = None
    if categorical_columns:
        columns = X.columns
        cat_features = np.where(columns.isin(categorical_columns))[0]

        for col in list(columns[cat_features]):
            X[col] = X[col].astype(str)

    gs, _ = load_graphs(f"{input_folder}/graph.dgl")
    graph = gs[0]

    with open(f"{input_folder}/masks.json") as f:
        masks = json.load(f)

    return graph, X, y, cat_features, masks


def normalize_features(X, train_mask, val_mask, test_mask):
    min_max_scaler = preprocessing.MinMaxScaler()
    A = X.to_numpy(copy=True)
    A[train_mask] = min_max_scaler.fit_transform(A[train_mask])
    A[val_mask + test_mask] = min_max_scaler.transform(A[val_mask + test_mask])
    return pd.DataFrame(A, columns=X.columns).astype(float)


def replace_na(X, train_mask):
    if X.isna().any().any():
        return X.fillna(X.iloc[train_mask].min() - 1)
    return X


def encode_cat_features(X, y, cat_features, train_mask, val_mask, test_mask):
    enc = CatBoostEncoder()
    A = X.to_numpy(copy=True)
    b = y.to_numpy(copy=True)
    A[np.ix_(train_mask, cat_features)] = enc.fit_transform(
        A[np.ix_(train_mask, cat_features)], b[train_mask]
    )
    A[np.ix_(val_mask + test_mask, cat_features)] = enc.transform(
        A[np.ix_(val_mask + test_mask, cat_features)]
    )
    A = A.astype(float)
    return pd.DataFrame(A, columns=X.columns)


""" For CPF"""


class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form."""

    def __init__(
        self,
        adj_matrix,
        attr_matrix=None,
        labels=None,
        node_names=None,
        attr_names=None,
        class_names=None,
        metadata=None,
    ):
        """Create an attributed graph.

        Parameters
        ----------
        adj_matrix : sp.csr_matrix, shape [num_nodes, num_nodes]
            Adjacency matrix in CSR format.
        attr_matrix : sp.csr_matrix or np.ndarray, shape [num_nodes, num_attr], optional
            Attribute matrix in CSR or numpy format.
        labels : np.ndarray, shape [num_nodes], optional
            Array, where each entry represents respective node's label(s).
        node_names : np.ndarray, shape [num_nodes], optional
            Names of nodes (as strings).
        attr_names : np.ndarray, shape [num_attr]
            Names of the attributes (as strings).
        class_names : np.ndarray, shape [num_classes], optional
            Names of the class labels (as strings).
        metadata : object
            Additional metadata such as text.

        """
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError(
                "Adjacency matrix must be in sparse format (got {0} instead)".format(
                    type(adj_matrix)
                )
            )

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError(
                    "Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)".format(
                        type(attr_matrix)
                    )
                )

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency and attribute matrices don't agree"
                )

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency matrix and the label vector don't agree"
                )

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency matrix and the node names don't agree"
                )

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError(
                    "Dimensions of the attribute matrix and the attribute names don't agree"
                )

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self):
        """Get the number of nodes in the graph."""
        return self.adj_matrix.shape[0]

    def num_edges(self):
        """Get the number of edges in the graph.

        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

    def get_neighbors(self, idx):
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx : int
            Index of the node whose neighbors are of interest.

        """
        return self.adj_matrix[idx].indices

    def is_directed(self):
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self):
        """Convert to an undirected graph (make adjacency matrix symmetric)."""
        if self.is_weighted():
            raise ValueError("Convert to unweighted graph first.")
        else:
            self.adj_matrix = self.adj_matrix + self.adj_matrix.T
            self.adj_matrix[self.adj_matrix != 0] = 1
        return self

    def is_weighted(self):
        """Check if the graph is weighted (edge weights other than 1)."""
        return np.any(np.unique(self.adj_matrix[self.adj_matrix != 0].A1) != 1)

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    # Quality of life (shortcuts)
    def standardize(self):
        """Select the LCC of the unweighted/undirected/no-self-loop graph.

        All changes are done inplace.

        """
        G = self.to_unweighted().to_undirected()
        G.adj_matrix = eliminate_self_loops_adj(G.adj_matrix)
        G = largest_connected_components(G, 1)
        return G

    def unpack(self):
        """Return the (A, X, z) triplet."""
        return self.adj_matrix, self.attr_matrix, self.labels


def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.

    """
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix(
            (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
            shape=loader["adj_shape"],
        )

        if "attr_data" in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix(
                (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
                shape=loader["attr_shape"],
            )
        elif "attr_matrix" in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader["attr_matrix"]
        else:
            attr_matrix = None

        if "labels_data" in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix(
                (
                    loader["labels_data"],
                    loader["labels_indices"],
                    loader["labels_indptr"],
                ),
                shape=loader["labels_shape"],
            )
        elif "labels" in loader:
            # Labels are stored as a numpy array
            labels = loader["labels"]
        else:
            labels = None

        node_names = loader.get("node_names")
        attr_names = loader.get("attr_names")
        class_names = loader.get("class_names")
        metadata = loader.get("metadata")

    return SparseGraph(
        adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata
    )


def sample_per_class(
    random_state, labels, num_examples_per_class, forbidden_indices=None
):
    """
    Used in get_train_val_test_split, when we try to get a fixed number of examples per class
    """

    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [
            random_state.choice(
                sample_indices_per_class[class_index],
                num_examples_per_class,
                replace=False,
            )
            for class_index in range(len(sample_indices_per_class))
        ]
    )


def get_train_val_test_split(
    random_state,
    labels,
    train_examples_per_class=None,
    val_examples_per_class=None,
    test_examples_per_class=None,
    train_size=None,
    val_size=None,
    test_size=None,
):

    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))
    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False
        )

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state,
            labels,
            val_examples_per_class,
            forbidden_indices=train_indices,
        )
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(
            random_state,
            labels,
            test_examples_per_class,
            forbidden_indices=forbidden_indices,
        )
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert (
            len(np.concatenate((train_indices, val_indices, test_indices)))
            == num_samples
        )

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices
