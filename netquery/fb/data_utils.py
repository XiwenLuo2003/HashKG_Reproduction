import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from collections import defaultdict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from netquery.graph import Graph, Query, _reverse_edge

def load_graph_and_ontology(data_dir, embed_dim):
    # Load graph data
    with open(os.path.join(data_dir, 'graph_data.pkl'), 'rb') as f:
        rels, adj_lists, node_maps = pickle.load(f)
        
    num_entities = len(node_maps['entity'])
    
    # Load type2domain
    type2domain = {}
    with open(os.path.join(data_dir, 'type2domain.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                type2domain[int(parts[0])] = int(parts[1])
                
    num_types = max(type2domain.keys()) + 1 if type2domain else 0
    num_domains = max(type2domain.values()) + 1 if type2domain else 0

    # Read entityType.txt to find max_types
    max_types = 0
    with open(os.path.join(data_dir, 'entityType.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            num_t = int(parts[1])
            max_types = max(max_types, num_t)
            
    # Pad arrays: index num_entities is reserved for padding (node -1)
    entity_types = np.zeros((num_entities + 1, max_types), dtype=np.int64)
    entity_weights = np.zeros((num_entities + 1, max_types), dtype=np.float32)
    entity_domains = np.zeros((num_entities + 1, max_types), dtype=np.int64)
    
    with open(os.path.join(data_dir, 'entityType.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            e_id = int(parts[0])
            if e_id >= num_entities:
                continue
            num_t = int(parts[1])
            
            for i in range(num_t):
                t_id = int(parts[2 + 2*i])
                w = float(parts[3 + 2*i])
                entity_types[e_id, i] = t_id
                entity_weights[e_id, i] = w
                entity_domains[e_id, i] = type2domain.get(t_id, 0)
                
    ontology_data = {
        'num_types': num_types,
        'num_domains': num_domains,
        'entity_types': torch.LongTensor(entity_types),
        'entity_weights': torch.FloatTensor(entity_weights),
        'entity_domains': torch.LongTensor(entity_domains),
        'num_entities': num_entities
    }
    
    # Process node_maps
    node_maps_dict = {m: {n: i for i, n in enumerate(id_list)} for m, id_list in node_maps.items()}
    for m in node_maps_dict:
        node_maps_dict[m][-1] = -1
        
    feature_dims = {m: embed_dim for m in rels}
    
    # Base embeddings for entities (index 0 is for padding -1)
    base_embeddings = nn.Embedding(num_entities + 1, embed_dim)
    base_embeddings.weight.data.normal_(0, 1./embed_dim)
    
    feature_modules = {'entity': base_embeddings}
    
    # features function
    def features_func(nodes, mode):
        mapped = [node_maps_dict[mode][n] + 1 for n in nodes]
        return feature_modules[mode](torch.autograd.Variable(torch.LongTensor(mapped)))
        
    graph = Graph(features_func, feature_dims, rels, adj_lists)
    
    return graph, feature_modules, node_maps_dict, ontology_data

def cudify(feature_modules, ontology_data=None):
    if ontology_data is not None:
        ontology_data['entity_types'] = ontology_data['entity_types'].cuda()
        ontology_data['entity_weights'] = ontology_data['entity_weights'].cuda()
        ontology_data['entity_domains'] = ontology_data['entity_domains'].cuda()

def make_fb15k_train_test_query_data(data_dir, num_workers=80, train_samples_per_worker=12500, test_samples_per_worker=1250):
    from netquery.data_utils import parallel_sample
    print("Loading graph for complex query sampling...")
    graph, _, _, _ = load_graph_and_ontology(data_dir, 10)
    
    print(f"Starting parallel sampling for training queries with {num_workers} workers...")
    queries_2, queries_3 = parallel_sample(graph, num_workers, train_samples_per_worker, data_dir, test=False)
    
    print(f"Starting parallel sampling for testing queries with {num_workers} workers...")
    t_queries_2, t_queries_3 = parallel_sample(graph, num_workers, test_samples_per_worker, data_dir, test=True)
    
    # Filter out test queries that are present in training queries
    t_queries_2 = list(set(t_queries_2) - set(queries_2))
    t_queries_3 = list(set(t_queries_3) - set(queries_3))
    
    print("Saving sampled query data to pickle files...")
    pickle.dump([q.serialize() for q in queries_2], open(os.path.join(data_dir, "train_queries_2.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in queries_3], open(os.path.join(data_dir, "train_queries_3.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    
    # Split the testing samples into validation and test sets (half and half)
    mid_2 = len(t_queries_2) // 2
    mid_3 = len(t_queries_3) // 2
    
    pickle.dump([q.serialize() for q in t_queries_2[:mid_2]], open(os.path.join(data_dir, "val_queries_2.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in t_queries_3[:mid_3]], open(os.path.join(data_dir, "val_queries_3.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in t_queries_2[mid_2:]], open(os.path.join(data_dir, "test_queries_2.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in t_queries_3[mid_3:]], open(os.path.join(data_dir, "test_queries_3.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Complex query sampling completed successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../fb15k_data")
    parser.add_argument("--num_workers", type=int, default=80)
    args = parser.parse_args()
    
    make_fb15k_train_test_query_data(args.data_dir, num_workers=args.num_workers)
