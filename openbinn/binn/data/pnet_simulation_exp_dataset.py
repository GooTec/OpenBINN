import os
import time
import copy
import gzip
import logging
import pickle
import json
import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional

import torch
from torch.utils.data import Dataset

cached_data = {}

class PnetSimExpDataSet(Dataset):
    """Experimental dataset for prostate cancer simulations."""

    def __init__(
        self,
        num_features: int = 3,
        root: Optional[str] = "./data/prostate/",
        valid_ratio: float = 0.102,
        test_ratio: float = 0.102,
        valid_seed: int = 0,
        test_seed: int = 7357,
    ):
        self.num_features = num_features
        self.root = root
        self._files = {}
        all_data, response = data_reader(filename_dict=self.raw_file_names, graph=False)
        self.subject_id = list(response.index)
        self.x = torch.tensor(all_data.to_numpy(), dtype=torch.float32)
        self.x = self.x.view(len(self.x), -1, self.num_features)
        self.y = torch.tensor(response.to_numpy(), dtype=torch.float32)

        self.genes = [g[0] for g in list(all_data.head(0))[0::self.num_features]]
        self.node_index = list(all_data.columns.get_level_values(0).unique())

        self.num_samples = len(self.y)
        self.num_test_samples = int(test_ratio * self.num_samples)
        self.num_valid_samples = int(valid_ratio * self.num_samples)
        self.num_train_samples = self.num_samples - self.num_test_samples - self.num_valid_samples
        self.split_index_by_rng(test_seed=test_seed, valid_seed=valid_seed)

    def split_index_by_rng(self, test_seed, valid_seed):
        rng_test = np.random.default_rng(test_seed)
        rng_valid = np.random.default_rng(valid_seed)
        test_split_perm = rng_test.permutation(self.num_samples)
        self.test_idx = list(test_split_perm[: self.num_test_samples])
        self.trainvalid_indices = test_split_perm[self.num_test_samples :]
        valid_split_perm = rng_valid.permutation(len(self.trainvalid_indices))
        self.valid_idx = list(self.trainvalid_indices[valid_split_perm[: self.num_valid_samples]])
        self.train_idx = list(self.trainvalid_indices[valid_split_perm[self.num_valid_samples :]])

    def split_index_by_file(self, train_fp, valid_fp, test_fp):
        train_set = pd.read_csv(train_fp, index_col=0)
        valid_set = pd.read_csv(valid_fp, index_col=0)
        test_set = pd.read_csv(test_fp, index_col=0)

        def _get_ids(df):
            if "id" in df.columns:
                return df["id"]
            for c in df.columns:
                if c.lower().startswith("id"):
                    return df[c]
            return df.index

        patients_train = list(_get_ids(train_set))
        both = set(self.subject_id).intersection(patients_train)
        self.train_idx = [self.subject_id.index(x) for x in both]

        patients_valid = list(_get_ids(valid_set))
        both = set(self.subject_id).intersection(patients_valid)
        self.valid_idx = [self.subject_id.index(x) for x in both]

        patients_test = list(_get_ids(test_set))
        both = set(self.subject_id).intersection(patients_test)
        self.test_idx = [self.subject_id.index(x) for x in both]

        assert len(self.train_idx) == len(set(self.train_idx))
        assert len(self.valid_idx) == len(set(self.valid_idx))
        assert len(self.test_idx) == len(set(self.test_idx))
        assert len(set(self.train_idx).intersection(set(self.valid_idx))) == 0
        assert len(set(self.train_idx).intersection(set(self.test_idx))) == 0
        assert len(set(self.valid_idx).intersection(set(self.test_idx))) == 0

    def __repr__(self):
        return f"PnetDataset(len={len(self)})"

    @property
    def raw_file_names(self):
        return {
            "selected_genes": os.path.join(self.root, self._files.get("selected_genes", "selected_genes.csv")),
            "response": os.path.join(self.root, self._files.get("response", "response.csv")),
            "mut_important": os.path.join(self.root, self._files.get("mut_important", "somatic_mutation_paper.csv")),
        }

    @property
    def processed_file_names(self):
        return "data-processed.pt"

    @property
    def processed_dir(self) -> str:
        return self.root

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x_sample = self.x[idx]
        y_sample = self.y[idx]
        sample_id = str(self.subject_id[idx])
        return x_sample, y_sample, sample_id


def data_reader(filename_dict, graph=True):
    assert "response" in filename_dict
    fd = copy.deepcopy(filename_dict)
    if graph:
        for f in filename_dict.values():
            if not os.path.isfile(f):
                raise FileNotFoundError(f)
        edge_dict = graph_reader_and_processor(graph_file=fd.pop("graph_file"))

    selected_genes = fd.pop("selected_genes")
    if selected_genes is not None:
        selected_genes = pd.read_csv(selected_genes)["genes"]
    use_coding_genes_only = None
    labels = get_response(fd.pop("response"))
    x_list = []
    y_list = []
    rows_list = []
    cols_list = []
    data_type_list = []
    for data_type, filename in fd.items():
        x, y, info, genes = load_data(filename=filename, selected_genes=selected_genes)
        x = processor(x, data_type)
        x_list.append(x)
        y_list.append(y)
        rows_list.append(info)
        cols_list.append(genes)
        data_type_list.append(data_type)
    res = combine(x_list, y_list, rows_list, cols_list, data_type_list, combine_type="union", use_coding_genes_only=use_coding_genes_only)
    all_data = res[0]
    response = labels.loc[all_data.index]
    if graph:
        return all_data, response, edge_dict
    else:
        return all_data, response


def graph_reader_and_processor(graph_file):
    graph_noext, _ = os.path.splitext(graph_file)
    graph_pickle = graph_noext + ".pkl"
    start_time = time.time()
    if os.path.exists(graph_pickle):
        with open(graph_pickle, "rb") as f:
            edge_dict = pickle.load(f)
    else:
        edge_dict = defaultdict(dict)
        with gzip.open(graph_file, "rt") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0, os.SEEK_SET)
            pbar = tqdm.tqdm(total=file_size, unit_scale=True, unit_divisor=1024, mininterval=1.0, desc="gene graph")
            for line in f:
                pbar.update(len(line))
                elems = line.strip().split("\t")
                if len(elems) == 0:
                    continue
                assert len(elems) == 3
                edge_dict[elems[0]][elems[1]] = float(elems[2])
                edge_dict[elems[1]][elems[0]] = float(elems[2])
            pbar.close()
        t0 = time.time()
        print("Caching the graph as a pickle...", end=None)
        with open(graph_pickle, "wb") as f:
            pickle.dump(edge_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f" done (took {time.time() - t0:.2f} seconds).")
    print(f"loading gene graph took {time.time() - start_time:.2f} seconds.")
    return edge_dict


def processor(x, data_type):
    if data_type == "mut_important":
        x[x > 1.0] = 1.0
    else:
        raise TypeError("unknown data type '%s' % data_type")
    return x


def get_response(response_filename):
    logging.info("loading response from %s" % response_filename)
    labels = pd.read_csv(response_filename)
    labels = labels.set_index("id")
    if "response" in cached_data:
        logging.warning("response in cached_data is being overwritten by '%s'" % response_filename)
    else:
        logging.warning("response in cached_data is being set by '%s'" % response_filename)
    cached_data["response"] = labels
    return labels


def load_data(filename, response=None, selected_genes=None):
    logging.info("loading data from %s," % filename)
    if filename in cached_data:
        logging.info("loading from memory cached_data")
        data = cached_data[filename]
    else:
        data = pd.read_csv(filename, index_col=0)
        cached_data[filename] = data
    logging.info(data.shape)
    if response is None:
        if "response" in cached_data:
            logging.info("loading from memory cached_data")
            labels = cached_data["response"]
        else:
            raise ValueError("abort: must read response first, but can't find it in cached_data")
    else:
        labels = copy.deepcopy(response)
    all = data.join(labels, how="inner")
    all = all[~all["response"].isnull()]
    response = all["response"]
    samples = all.index
    del all["response"]
    x = all
    genes = all.columns
    if not selected_genes is None:
        intersect = list(set.intersection(set(genes), selected_genes))
        if len(intersect) < len(selected_genes):
            logging.warning("some genes don't exist in the original data set")
        x = x.loc[:, intersect]
        genes = intersect
    logging.info("loaded data %d samples, %d variables, %d responses " % (x.shape[0], x.shape[1], response.shape[0]))
    logging.info(len(genes))
    return x, response, samples, genes


def combine(
    x_list,
    y_list,
    rows_list,
    cols_list,
    data_type_list,
    combine_type,
    use_coding_genes_only=None,
):
    cols_list_set = [set(list(c)) for c in cols_list]
    if combine_type == "intersection":
        cols = set.intersection(*cols_list_set)
    else:
        cols = set.union(*cols_list_set)
    logging.debug("step 1 union of gene features", len(cols))
    if use_coding_genes_only is not None:
        assert os.path.isfile(use_coding_genes_only), "you specified a filepath to filter coding genes, but the file doesn't exist"
        f = os.path.join(use_coding_genes_only)
        coding_genes_df = pd.read_csv(f, sep="\t", header=None)
        coding_genes_df.columns = ["chr", "start", "end", "name"]
        coding_genes = set(coding_genes_df["name"].unique())
        cols = cols.intersection(coding_genes)
        logging.debug("step 2 intersect w/ coding", len(coding_genes), "; coding AND in cols", len(cols))
    all_cols = list(cols)
    all_cols_df = pd.DataFrame(index=all_cols)
    df_list = []
    for x, y, r, c, d in zip(x_list, y_list, rows_list, cols_list, data_type_list):
        df = pd.DataFrame(x, columns=c, index=r)
        df = df.T.join(all_cols_df, how="right")
        df = df.T
        logging.info("step 3 fill NA-%s num NAs=" % d, df.isna().sum().sum())
        df = df.fillna(0)
        df_list.append(df)
    all_data = pd.concat(df_list, keys=data_type_list, join="inner", axis=1)
    all_data = all_data.swaplevel(i=0, j=1, axis=1)
    order = sorted(all_data.columns.levels[0])
    all_data = all_data.reindex(columns=order, level=0)
    x = all_data.values
    reordering_df = pd.DataFrame(index=all_data.index)
    y = reordering_df.join(y, how="left")
    y = y.values
    cols = all_data.columns
    rows = all_data.index
    logging.debug("After combining, loaded data %d samples, %d variables, %d responses " % (x.shape[0], x.shape[1], y.shape[0]))
    return all_data, x, y, rows, cols
