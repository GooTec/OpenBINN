import networkx as nx
import re
import itertools
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Sequence
from .reactome import Reactome

class ReactomeNetwork:
    def __init__(self, reactome_kws):
        self.reactome = Reactome(**reactome_kws)
        self.netx = self.get_reactome_networkx()

    def get_terminals(self):
        return [n for n, d in self.netx.out_degree() if d == 0]

    def get_roots(self):
        return get_nodes_at_level(self.netx, distance=1)

    def get_reactome_networkx(self) -> nx.Graph:
        hierarchy = self.reactome.hierarchy
        human_hierarchy = hierarchy[hierarchy["child"].str.contains("HSA")]
        net = nx.from_pandas_edgelist(human_hierarchy, "child", "parent", create_using=nx.DiGraph())
        net.name = "reactome"
        roots = [n for n, d in net.in_degree() if d == 0]
        root_node = "root"
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)
        return net

    def get_tree(self):
        G = nx.bfs_tree(self.netx, "root")
        return G

    def get_completed_network(self, n_levels: int) -> nx.Graph:
        G = complete_network(self.netx, n_levels=n_levels)
        return G

    def get_completed_tree(self, n_levels: int) -> nx.Graph:
        G = self.get_tree()
        G = complete_network(G, n_levels=n_levels)
        return G

    def get_layers(self, n_levels: int, direction: str = "root_to_leaf") -> List[Dict[str, List[str]]]:
        if direction == "root_to_leaf":
            net = nx.bfs_tree(self.netx, "root")
            layers = get_layers_from_net(net, n_levels)
        else:
            net = self.get_completed_network(5)
            layers = get_layers_from_net(net, 5)
            layers = layers[5 - n_levels : 5]
        terminal_nodes = [n for n, d in self.netx.out_degree() if d == 0]
        genes_df = self.reactome.pathway_genes
        d = {}
        missing_pathways = []
        for p in terminal_nodes:
            pathway_name = re.sub("_copy.*", "", p)
            genes = genes_df[genes_df["group"] == pathway_name]["gene"].unique()
            if len(genes) == 0:
                missing_pathways.append(pathway_name)
            d[pathway_name] = genes
        layers.append(d)
        return layers


def get_layer_maps(
    reactome: ReactomeNetwork,
    genes,
    n_levels: int,
    direction: str,
    add_unk_genes: bool = False,
    verbose: bool = False,
) -> List[pd.DataFrame]:
    reactome_layers = reactome.get_layers(n_levels, direction)
    filtering_index = sorted(genes)
    maps = []
    for i, layer in enumerate(reactome_layers[::-1]):
        if verbose:
            print("layer #", i)
        crt_map = get_map_from_layer(layer)
        filter_df = pd.DataFrame(index=filtering_index)
        if verbose:
            print("filtered_map", filter_df.shape)
        filtered_map = filter_df.merge(crt_map, right_index=True, left_index=True, how="left")
        if add_unk_genes:
            filtered_map["UNK"] = 0
            ind = filtered_map.sum(axis=1) == 0
            filtered_map.loc[ind, "UNK"] = 1
        filtered_map = filtered_map.fillna(0)
        filtering_index = filtered_map.columns
        if verbose:
            logging.info("layer {} , # of edges  {}".format(i, filtered_map.sum().sum()))
        filtered_map = filtered_map[sorted(filtered_map.columns)]
        filtered_map = filtered_map.loc[sorted(filtered_map.index)]
        maps.append(filtered_map)
    return maps


def get_map_from_layer(layer_dict: Dict[str, Sequence[str]]) -> pd.DataFrame:
    pathways = list(layer_dict.keys())
    genes = sorted(list(np.unique(list(itertools.chain.from_iterable(layer_dict.values())))))
    n_pathways = len(pathways)
    n_genes = len(genes)
    mat = np.zeros((n_pathways, n_genes))
    for p, gs in layer_dict.items():
        g_inds = [genes.index(g) for g in gs]
        p_ind = pathways.index(p)
        mat[p_ind, g_inds] = 1
    df = pd.DataFrame(mat, index=pathways, columns=genes)
    return df.T


def add_edges(G: nx.Graph, node: str, n_levels: int) -> nx.Graph:
    edges = []
    source = node
    for l in range(n_levels):
        target = node + "_copy" + str(l + 1)
        edge = (source, target)
        source = target
        edges.append(edge)
    G.add_edges_from(edges)
    return G


def complete_network(G: nx.Graph, n_levels: int = 4) -> nx.Graph:
    sub_graph = nx.ego_graph(G, "root", radius=n_levels)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source="root", target=node))
        if distance <= n_levels:
            diff = n_levels - distance + 1
            sub_graph = add_edges(sub_graph, node, diff)
    return sub_graph


def get_nodes_at_level(net: nx.Graph, distance: int) -> List[str]:
    nodes = set(nx.ego_graph(net, "root", radius=distance))
    if distance >= 1:
        nodes -= set(nx.ego_graph(net, "root", radius=distance - 1))
    return list(nodes)


def get_layers_from_net(net: nx.Graph, n_levels: int) -> List[Dict[str, List[str]]]:
    layers = []
    for i in range(n_levels):
        nodes = get_nodes_at_level(net, i)
        d = {}
        for n in nodes:
            n_name = re.sub("_copy.*", "", n)
            nexts = net.successors(n)
            d[n_name] = [re.sub("_copy.*", "", nex) for nex in nexts]
        layers.append(d)
    return layers
