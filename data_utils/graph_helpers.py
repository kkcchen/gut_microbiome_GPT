import pandas as pd
import numpy as np
from collections import defaultdict
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from collections import deque

from torch import nn
import anndata as ad


# --- helper tests ---
def assign_categories(taxa_name):
    """
    Assign generic categories for taxa names by splitting by '.'
    Handles standard 7-level taxonomy (Domain to Genus/Species).
    """
    split_taxa = taxa_name.split(".")
    # Standard ranks used in this project
    categories = ["Domain", "Phylum", "Class", "Order", "Family", "Genus"]
    
    # If we have more or fewer than 6, we just take the first 6 or pad
    res = split_taxa[:6]
    while len(res) < 6:
        res.append(np.nan)
    return res


def is_invalid_name(name):
    INVALID_TOKENS = {"na", "n/a", "unknown", ""}  # lowercased
    INVALID_SUBSTRINGS = ["incertae sedis", "unknown"]
    """Return True for names we want to remove (NA, Incertae Sedis, etc)."""
    if pd.isna(name):
        return True
    s = str(name).strip()
    if s == "":
        return True
    low = s.lower()
    if low in INVALID_TOKENS:
        return True
    for sub in INVALID_SUBSTRINGS:
        if sub in low:
            return True
    return False


def get_valid_name(name_array):
    valid_chain = []
    distances = []
    end_invalid = False
    
    current_distance = 1
    name_array = np.array(name_array)
    for i, name in enumerate(name_array):
        if pd.isna(name):
            current_distance += 1
            continue
        name = str(name).strip()
        if is_invalid_name(name):
            # check if invalid name is surrounded by valid names
            # if i > 0 and i < len(name_array) - 1:
            #     prev_name = name_array[i - 1]
            #     next_name = name_array[i + 1]
            #     if (
            #         not pd.isna(prev_name)
            #         and not is_invalid_name(str(prev_name).strip())
            #         and not pd.isna(next_name)
            #         and not is_invalid_name(str(next_name).strip())
            #     ):
            #         print("Invalid name surrounded by valids:", name_array)
            current_distance += 1
            continue
        else:
            valid_chain.append(name)
            distances.append(current_distance)
            current_distance = 1

    if valid_chain:
        last_name = name_array[-1]
        if is_invalid_name(last_name):
            end_invalid = True

    return valid_chain, distances[1:], end_invalid


# --- build initial parent->children mapping from taxon dataframe ---
def build_parent_children(taxon_df):
    parent_children = defaultdict(set)
    child_parents = defaultdict(set)
    parent_child_distances = defaultdict(set)
    ranks = taxon_df.columns.tolist()
    duplicate_nodenames = {}
    do_not_collapse = set()
    isolated_nodes = set()

    for _, row in taxon_df.iterrows():
        valid_chain, distances, end_invalid = get_valid_name(row[ranks].tolist())
        if end_invalid:
            do_not_collapse.add(valid_chain[-1]) # if ends with invalid, do not collapse the last valid node
        
        if len(valid_chain) == 1:
            # single valid name, no parent-child relationship to add
            isolated_nodes.add(valid_chain[0])
        # Connect consecutive valid taxa directly
        for i in range(len(valid_chain) - 1):
            parent = valid_chain[i]
            child = valid_chain[i + 1]
            if parent != child:
                if child in child_parents and parent not in child_parents[child]:
                    sub_name = f"{parent}|{child}"
                    assert sub_name not in child_parents, f"{sub_name} was in child_parents"
                    child_parents[sub_name].add(parent)
                    parent_children[parent].add(sub_name)
                    duplicate_nodenames[(parent, child)] = sub_name
                    parent_child_distances[(parent, sub_name)] = distances[i]
                else:
                    parent_children[parent].add(child)
                    child_parents[child].add(parent)
                    parent_child_distances[(parent, child)] = distances[i]



    # Sort keys for readability
    parent_children = {k: parent_children[k] for k in sorted(parent_children)}
    child_parents = {k: child_parents[k] for k in sorted(child_parents)}
    return dict(parent_children), dict(child_parents), dict(parent_child_distances), duplicate_nodenames, do_not_collapse, isolated_nodes

# --- collapse single-child nodes as described:
# if a node P has exactly one child C, remove C and connect P -> grandchildren_of_C
def collapse_single_child_nodes(parent_children, child_parents, parent_child_distances, roots, do_not_collapse):
    queue = deque(roots)
    while queue:
        parent = queue.popleft()
        assert (parent in roots) or (len(child_parents[parent]) == 1), "the parent should have exactly one parent unless it's a root"
        
        if parent not in parent_children: # leaf node
            continue
        # if the parent has exactly one child, we can collapse
        if len(parent_children[parent]) == 1 and parent not in roots and parent not in do_not_collapse:
            grandparent = next(iter(child_parents[parent]))
            child = next(iter(parent_children[parent]))
            
            # get new name for child
            child_newname = f"{parent}.{child}"
            assert child_newname not in parent_children, f"Duplicate node name: {child_newname}"

            # connect grandparent -> child
            parent_children[grandparent].add(child_newname)
            parent_children[grandparent].discard(parent)
            
            child_parents[child_newname] = {grandparent}
            del child_parents[child]
            
            del parent_children[parent]
            del child_parents[parent]

            if child in parent_children: # if child is not a leaf, reconnect grandchild references
                parent_children[child_newname] = parent_children[child]
                for gc in parent_children[child]:
                    child_parents[gc].add(child_newname)
                    child_parents[gc].discard(child)
                    
                    # also reconnect distance references
                    parent_child_distances[(child_newname, gc)] = parent_child_distances[(child, gc)]
                    del parent_child_distances[(child, gc)]
                del parent_children[child]
                
                
            
            if child in do_not_collapse:
                do_not_collapse.add(child_newname)
                do_not_collapse.discard(child)
            
            # update distances
            gp_p_distance = parent_child_distances[(grandparent, parent)]
            p_c_distance = parent_child_distances[(parent, child)]
            parent_child_distances[(grandparent, child_newname)] = gp_p_distance + p_c_distance
            del parent_child_distances[(grandparent, parent)]
            del parent_child_distances[(parent, child)]
            
            queue.append(child_newname)
        else:
            queue.extend(parent_children[parent])

    return parent_children, child_parents, parent_child_distances


def get_leaf_embedding_index(path_parts, data, return_last_valid=False):
    """
    Check if a taxonomic path (list of names) exists in the current graph.
    Handles merged nodes like 'Bacteria.Bacillota'.
    Skips invalid names like 'NA' or 'Incertae Sedis'.
    """
    # Remove invalid taxa first
    parts = [p for p in path_parts if not is_invalid_name(p)]
    
    
    if len(parts) < 1:
        return data.root_index  # all names invalid, return root index
    if len(parts) == 1:
        # return parts[0] in data.names
        if parts[0] not in data.name_to_idx:
            return None
        return data.name_to_idx[parts[0]]

    # Traverse pairwise through the path
    i = 0
    used_merge_name = False
    while i < len(parts) - 1:
        found_edge = False
        p = parts[i] if not used_merge_name else merged_name
        c = parts[i + 1]
        
        # check in data.duplicate_nodenames for substitutions
        if (p, c) in data.duplicate_nodenames:
            c = data.duplicate_nodenames[(p, c)]

        # Case 1: exact match parent → child
        if p in data.parent_children:
            if c in data.parent_children[p]:
                found_edge = True
                used_merge_name = False
            else:
                # Try progressively merged names:
                # e.g., if "Bacteria" and "Bacillota" are merged as "Bacteria.Bacillota" we should match it.
                j = i + 2
                merged_name = c
                while j < len(parts):
                    merged_candidate = ".".join(parts[i+1:j+1])
                    if merged_candidate in data.names:
                        merged_name = merged_candidate
                        found_edge = True
                        used_merge_name = True
                        break
                    else:
                        j += 1

                # Advance to the deepest merged node we found
                c = merged_name
                i = j - 1  # because next loop iteration starts from here
            
        if not found_edge:
            if return_last_valid:
                return data.name_to_idx[p]
            return None

        i += 1

    return data.name_to_idx[c]

# --- produce torch_geometric Data ---
def build_tg_data_from_taxon_df(taxon_df, vocab_list, undirected: bool = True):
    """
    Constructs a PyTorch Geometric `Data` object from a taxonomic DataFrame.
    This function processes a taxonomic DataFrame to build a graph representation
    of the taxonomy. It creates nodes, edges, and embeddings for the graph, and
    attaches additional metadata for debugging and mapping purposes.
    Args:
        taxon_df (pd.DataFrame): A DataFrame containing taxonomic information. 
            Each row represents a taxonomic entity, and columns represent hierarchical levels.
        embedding_dim (int): The dimensionality of the node embeddings.
        vocab_list (list): A list of vocabulary terms in the order of the vocabulary. 
            Each term must correspond to a leaf node in the taxonomy.
        undirected (bool, optional): If True, the graph edges will be undirected. 
            Defaults to True.
    Returns:
        torch_geometric.data.Data: A PyTorch Geometric `Data` object containing:
            - `edge_index` (torch.Tensor): The edge list of the graph.
            - `names` (list): List of node names.
            - `name_to_idx` (dict): Mapping from node names to their indices.
            - `parent_children` (dict): Mapping from parent nodes to their child nodes.
            - `child_parents` (dict): Mapping from child nodes to their parent nodes.
            - `duplicate_nodenames` (list): List of duplicate node names, if any.
            - `vocabindex_to_nodeindex` (torch.Tensor): Mapping from vocabulary indices 
              to node indices in the graph.
    Raises:
        AssertionError: If a vocabulary term in `vocab_list` cannot be mapped to a node in the graph.
    Notes:
        - The `vocab_list` must be in the same order as the vocabulary used in downstream tasks.
        - The function validates the tree structure to ensure that each child node has only one parent.
        - Single-child nodes in the taxonomy are collapsed for simplicity.
    """
    
    
    parent_children, child_parents, parent_child_distances, sub_names, do_not_collapse, isolated_nodes = build_parent_children(taxon_df)
    # Validate tree structure: child_parents values should all have length one
    for child, parents in child_parents.items():
        if len(parents) > 1:
            print(f"Tree structure violated: Child '{child}' has multiple parents: {parents}")
    
    # collapse single-child nodes
    roots = [n for n in parent_children.keys() if n not in child_parents]
    print(f"Identified {roots} root nodes (no parents).")
    parent_children, child_parents, parent_child_distances = collapse_single_child_nodes(parent_children, child_parents, parent_child_distances, roots, do_not_collapse)

    # add root node to connect "roots" if multiple
    if len(roots) > 1:
        root_name = "Life"
        assert root_name not in parent_children, f"Root name {root_name} already exists in parent_children"
        roots = set(roots).union(isolated_nodes)
        parent_children[root_name] = roots
        for r in roots:
            child_parents[r] = {root_name}
            # add distances
            parent_child_distances[(root_name, r)] = 1
    else:
        root_name = roots[0]
        
    # build final node set and edges
    nodes = set(parent_children.keys()) | set(child_parents.keys())

    nodes = sorted(nodes)  # deterministic ordering
    name_to_idx = {n: i for i, n in enumerate(nodes)}

    print(len(parent_child_distances))

    src = []
    dst = []
    edge_dists = []
    for (p, c), dist in parent_child_distances.items():
        p_idx = name_to_idx[p]
        c_idx = name_to_idx[c]
        src.append(p_idx)
        dst.append(c_idx)
        edge_dists.append(dist)
    
    edge_dists = torch.tensor(edge_dists, dtype=torch.float).unsqueeze(1)
        
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    if undirected:
        edge_index, edge_dists = to_undirected(edge_index, edge_dists)
    
    data = Data(edge_index=edge_index, edge_attr=edge_dists)
    
    # attach names (useful for debugging / mapping back)
    data.names = nodes
    data.num_nodes = len(nodes)
    data.name_to_idx = name_to_idx
    data.parent_children = parent_children
    data.child_parents = child_parents
    data.duplicate_nodenames = sub_names
    data.root_index = name_to_idx[root_name]
    
    data.parent_child_distances = parent_child_distances
    
    data.vocabindex_to_nodeindex = torch.zeros(len(vocab_list), dtype=torch.long)
    for vocab_index, fullname in vocab_list.items():
        separated_name = taxon_df.loc[fullname]
        data.vocabindex_to_nodeindex[vocab_index] = get_leaf_embedding_index(separated_name.to_list(), data)
        assert data.vocabindex_to_nodeindex[vocab_index] is not None, f"Could not find node for {fullname}"
    
    return data


def split_taxonomy_and_save(data, exceptions_path):    
    # Split the taxa names by "." and count the maximum number of levels
    taxa_split = data.var['taxa'].str.split('.')
    print(taxa_split.apply(len).value_counts().sort_index())
    
    # exceptions located in the period_taxa.csv file
    period_taxa = pd.read_csv(exceptions_path, header=None)
    period_taxa = period_taxa.drop(index=0)

    # Merge entries in each row with a '.' separator
    # Drop the first row and merge entries in each row with a '.' separator
    merged_taxa = period_taxa.apply(lambda row: '.'.join(row.dropna().astype(str)), axis=1)
    # Create a DataFrame with merged_taxa as the index and period_taxa as the data
    period_taxa.index = merged_taxa
    
    # Define a function to assign categories based on the period_taxa DataFrame
    def assign_categories(taxa_name, period_taxa):
        split_taxa = taxa_name.split(".")
        if len(split_taxa) > 6:
            # Use the period_taxa Series to assign categories properly
            if taxa_name not in period_taxa.index:
                raise ValueError(f"Taxa '{taxa_name}' not found in period_taxa.")
            return period_taxa.loc[taxa_name]
        else:
            # Assign generic categories for taxa with 6 or fewer levels
            return split_taxa

    # Apply the function to the taxa names and create a new varm
    taxon_lists = data.var["taxa"].apply(assign_categories, period_taxa=period_taxa)
    categories = ["Domain", "Phylum", "Class", "Order", "Family", "Genus"]

    taxon_df = pd.DataFrame(taxon_lists.tolist(), index=data.var_names, columns=categories)

    return taxon_df


if __name__ == "__main__":
    # -----------------------
    # Example usage:
    # -----------------------
    DATA_DIR = "../../datasets/up_to_date_sep10/"

    data = ad.read_h5ad(DATA_DIR + "pretrain.h5ad")
    taxon_df = split_taxonomy_and_save(data, DATA_DIR + "period_taxa.csv")

    data.varm['taxonomy'] = taxon_df
    # data.write_h5ad(DATA_DIR + "pretrain.h5ad")
    graph_data = build_tg_data_from_taxon_df(taxon_df)
    print(graph_data.leaf_embedding_indices)
    print("First 10 node names:", graph_data.names[:10])
