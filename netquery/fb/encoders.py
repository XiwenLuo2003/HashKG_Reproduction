import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class BaseHashEncoder(nn.Module):
    def __init__(self, features, feature_modules, beta=1):
        super(BaseHashEncoder, self).__init__()
        for name, module in feature_modules.items():
            self.add_module("feat-"+name, module)
        self.features = features
        self.beta = beta
        
    def tohash(self, x):
        return torch.sign(torch.sign(x).add(0.1))

class WHEEncoder(BaseHashEncoder):
    def __init__(self, features, feature_modules, ontology_data, embed_dim, cuda=False, beta=1):
        super(WHEEncoder, self).__init__(features, feature_modules, beta)
        
        self.num_types = ontology_data['num_types']
        self.num_entities = ontology_data['num_entities']
        
        self.entity_types = nn.Parameter(ontology_data['entity_types'], requires_grad=False)
        self.entity_weights = nn.Parameter(ontology_data['entity_weights'], requires_grad=False)
        
        # M_c for all types: shape [num_types, embed_dim, embed_dim]
        # We also need a dummy type matrix for padding at index num_types if needed
        # but type IDs range from 0 to num_types - 1. We'll add one more for padding (0 matrix)
        self.type_matrices = nn.Embedding(self.num_types + 1, embed_dim * embed_dim)
        init.xavier_uniform_(self.type_matrices.weight)
        # Zero out the padding type matrix
        with torch.no_grad():
            self.type_matrices.weight[self.num_types].fill_(0.0)
            
        self.embed_dim = embed_dim
        self.use_cuda = cuda

    def forward(self, nodes, mode, sign=False, offset=None, **kwargs):
        # 1. Get continuous base embeddings x_e
        # x_e shape: [batch_size, embed_dim, 1]
        x_e = self.features(nodes, mode).unsqueeze(-1)
        
        # Map nodes to indices in our tensor: -1 maps to self.num_entities
        nodes_mapped = [n if n != -1 else self.num_entities for n in nodes]
        nodes_tensor = torch.LongTensor(nodes_mapped)
        if self.use_cuda:
            nodes_tensor = nodes_tensor.cuda()
            
        # 2. Get types and weights
        # types shape: [batch_size, max_types]
        types = self.entity_types[nodes_tensor]
        weights = self.entity_weights[nodes_tensor]
        
        # Handle zero weights where no types exist. Map them to the padding type (self.num_types)
        types = torch.where(weights > 0, types, torch.tensor(self.num_types).to(types.device))
        
        # 3. Compute entity projection matrix M_e = sum(alpha_i * M_c_i)
        # M_t shape: [batch_size, max_types, embed_dim * embed_dim]
        M_t = self.type_matrices(types)
        
        # Reshape to [batch_size, max_types, embed_dim, embed_dim]
        M_t = M_t.view(-1, types.size(1), self.embed_dim, self.embed_dim)
        
        # weights shape: [batch_size, max_types, 1, 1]
        w = weights.view(-1, types.size(1), 1, 1)
        
        # M_e shape: [batch_size, embed_dim, embed_dim]
        M_e = torch.sum(M_t * w, dim=1)
        
        # 4. Project base embeddings: M_e * x_e
        # embeds shape: [batch_size, embed_dim]
        embeds = torch.bmm(M_e, x_e).squeeze(-1)
        # 确保输出总是 [batch_size, embed_dim]
        if len(embeds.size()) > 2:
            embeds = embeds.view(embeds.size(0), -1)
        
        # 5. Apply smooth continuation method
        ret = torch.tanh(self.beta * embeds)
        
        if sign:
            return self.tohash(ret)
        else:
            return ret

class RHEEncoder(BaseHashEncoder):
    def __init__(self, features, feature_modules, ontology_data, embed_dim, cuda=False, beta=1):
        super(RHEEncoder, self).__init__(features, feature_modules, beta)
        
        self.num_types = ontology_data['num_types']
        self.num_domains = ontology_data['num_domains']
        self.num_entities = ontology_data['num_entities']
        
        self.entity_types = nn.Parameter(ontology_data['entity_types'], requires_grad=False)
        self.entity_weights = nn.Parameter(ontology_data['entity_weights'], requires_grad=False)
        self.entity_domains = nn.Parameter(ontology_data['entity_domains'], requires_grad=False)
        
        self.type_matrices = nn.Embedding(self.num_types + 1, embed_dim * embed_dim)
        self.domain_matrices = nn.Embedding(self.num_domains + 1, embed_dim * embed_dim)
        
        init.xavier_uniform_(self.type_matrices.weight)
        init.xavier_uniform_(self.domain_matrices.weight)
        
        with torch.no_grad():
            self.type_matrices.weight[self.num_types].fill_(0.0)
            self.domain_matrices.weight[self.num_domains].fill_(0.0)
            
        self.embed_dim = embed_dim
        self.use_cuda = cuda

    def forward(self, nodes, mode, sign=False, offset=None, **kwargs):
        x_e = self.features(nodes, mode).unsqueeze(-1)
        
        nodes_mapped = [n if n != -1 else self.num_entities for n in nodes]
        nodes_tensor = torch.LongTensor(nodes_mapped)
        if self.use_cuda:
            nodes_tensor = nodes_tensor.cuda()
            
        types = self.entity_types[nodes_tensor]
        domains = self.entity_domains[nodes_tensor]
        weights = self.entity_weights[nodes_tensor]
        
        types = torch.where(weights > 0, types, torch.tensor(self.num_types).to(types.device))
        domains = torch.where(weights > 0, domains, torch.tensor(self.num_domains).to(domains.device))
        
        M_t = self.type_matrices(types).view(-1, types.size(1), self.embed_dim, self.embed_dim)
        M_d = self.domain_matrices(domains).view(-1, domains.size(1), self.embed_dim, self.embed_dim)
        
        # M_c = M_t * M_d (Recursive Hierarchical projection)
        M_c = torch.matmul(M_t, M_d)
        
        w = weights.view(-1, types.size(1), 1, 1)
        M_e = torch.sum(M_c * w, dim=1)
        
        embeds = torch.bmm(M_e, x_e).squeeze(-1)
        # 确保输出总是 [batch_size, embed_dim]
        if len(embeds.size()) > 2:
            embeds = embeds.view(embeds.size(0), -1)
            
        ret = torch.tanh(self.beta * embeds)
        
        if sign:
            return self.tohash(ret)
        else:
            return ret
