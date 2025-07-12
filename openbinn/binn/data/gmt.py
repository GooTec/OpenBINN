"""Functions for loading gene matrix transposed (GMT) files."""

import re
import pandas as pd


def load_gmt(filename: str, genes_col: int = 1, pathway_col: int = 0) -> pd.DataFrame:
    """Load a `.gmt` file into a DataFrame with columns `group` and `gene`."""
    data_dict_list = []
    with open(filename) as gmt:
        data_list = gmt.readlines()
        for row in data_list:
            elems = row.strip().split("\t")
            elems = [re.sub("_copy.*", "", g) for g in elems]
            elems = [re.sub("\\n.*", "", g) for g in elems]
            for gene in elems[genes_col:]:
                pathway = elems[pathway_col]
                data_dict_list.append({"group": pathway, "gene": gene})
    df = pd.DataFrame(data_dict_list)
    return df
