from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import GCNConv, GATConv, GINConv, GatedGraphConv, RGCNConv, RGATConv, GPSConv, AntiSymmetricConv
from torch_geometric.nn import global_add_pool as add_p, global_max_pool as max_p, global_mean_pool as mean_p
from torch.nn import Sequential, Linear, ReLU
from build_tree import build_stage_one_edges, build_stage_two_edges
from build_tree import build_diag_tree, build_atc_tree, build_proce_tree




class EHROntologyModel(nn.Module):
    """
    Encodes EHR graphs and computes similarity between two EHR graphs.
    """
    def __init__(self, config, vocab_emb, diag_voc, proce_voc, atc_voc, device):
        super(EHROntologyModel, self).__init__()

        # Embedding layers for ATC, Diagnosis, and Procedure ontologies
        self.atc_embedding = OntologyEmbedding(atc_voc, build_atc_tree,
                                               config.hidden_size, config.hidden_size,
                                               config.graph_heads)
        self.diag_embedding = OntologyEmbedding(diag_voc, build_diag_tree,
                                                config.hidden_size, config.hidden_size,
                                                config.graph_heads)
        self.proce_embedding = OntologyEmbedding(proce_voc, build_proce_tree,
                                                 config.hidden_size, config.hidden_size,
                                                 config.graph_heads)

        self.text_embedding = nn.Parameter(vocab_emb)
        glorot(self.text_embedding)

        self.device = device
        # 768 is BERT hidden size
        self.text_linear = nn.Linear(768, config.hidden_size)

        self.concat_emb_size = 2 * config.hidden_size

        self.out_channels = self.concat_emb_size
        self.gcn_conv_list = nn.ModuleList()
        self.use_conv = config.use_conv

        # Initialize graph convolutional layers based on configuration
        if config.use_conv == 'gin':
            self.gin_mlp = nn.Sequential(nn.Linear(self.concat_emb_size, self.concat_emb_size), nn.ReLU(
            ), nn.Linear(self.concat_emb_size, self.concat_emb_size))
            for _ in range(config.gcn_conv_nums-1):
                self.gcn_conv_list.append(GINConv(self.gin_mlp))
            self.gcn_conv_list.append(GINConv(nn.Sequential(nn.Linear(
                self.concat_emb_size, self.concat_emb_size), nn.ReLU(), nn.Linear(self.concat_emb_size, self.out_channels))))
        elif config.use_conv == 'rgcn':
            for _ in range(config.gcn_conv_nums - 1):
                self.gcn_conv_list.append(
                    RGCNConv(in_channels=self.concat_emb_size, out_channels=self.concat_emb_size, num_relations=9))
            self.gcn_conv_list.append(
                RGCNConv(in_channels=self.concat_emb_size, out_channels=self.out_channels, num_relations=9))
        elif config.use_conv == 'rgat':
            for _ in range(config.gcn_conv_nums - 1):
                self.gcn_conv_list.append(
                    RGATConv(in_channels=self.concat_emb_size, out_channels=self.concat_emb_size, num_relations=9))
            self.gcn_conv_list.append(
                RGATConv(in_channels=self.concat_emb_size, out_channels=self.out_channels, num_relations=9))
        elif config.use_conv == 'gps':
            channels = self.concat_emb_size
            for _ in range(config.gcn_conv_nums):
                linear = Sequential(
                    Linear(channels, channels),
                    ReLU(),
                    Linear(channels, channels),
                )
                self.gcn_conv_list.append(
                    GPSConv(channels, GINConv(linear), heads=2))
        elif config.use_conv == 'anti':
            channels = self.concat_emb_size
            for _ in range(config.gcn_conv_nums):
                self.gcn_conv_list.append(AntiSymmetricConv(channels))
        elif config.use_conv == 'gat':
            for _ in range(config.gcn_conv_nums-1):
                self.gcn_conv_list.append(GATConv(
                    in_channels=self.concat_emb_size, out_channels=self.concat_emb_size//config.num_heads, heads=config.num_heads))
            self.gcn_conv_list.append(GATConv(in_channels=self.concat_emb_size,
                                      out_channels=self.out_channels//config.num_heads, heads=config.num_heads))
        elif config.use_conv == 'gcn':
            for _ in range(config.gcn_conv_nums-1):
                self.gcn_conv_list.append(
                    GCNConv(in_channels=self.concat_emb_size, out_channels=self.concat_emb_size))
            self.gcn_conv_list.append(
                GCNConv(in_channels=self.concat_emb_size, out_channels=self.out_channels))
        elif config.use_conv == 'gated_conv':
            for _ in range(config.gcn_conv_nums-1):
                self.gcn_conv_list.append(GatedGraphConv(
                    out_channels=self.concat_emb_size, num_layers=1))
            self.gcn_conv_list.append(GatedGraphConv(
                out_channels=self.out_channels, num_layers=1))

        self.global_tensor_dim = self.out_channels
        self.neural_sim_layer = NeuralTensorModule(
            self.global_tensor_dim, config.pair_neurons)

        self.nodes_filter_layer = NodeFilterModule(self.out_channels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.pair_neurons, 2)

    # Multimodal Encoding Module
    def encode(self, all_embedding, x, graph_index, x_batch, edge_type=None):
        # Multimodal embedding for each code
        onto_x = all_embedding[x].squeeze(dim=1)
        text_x = self.text_linear(self.text_embedding[x].squeeze(dim=1))
        # Concatenate the multimodal embeddings
        x = torch.cat([onto_x, text_x], dim=-1)

        temp_x = x
        # Encoding the EHR graph
        if self.use_conv == 'rgcn' or self.use_conv == 'rgat':
            for gcn_conv in self.gcn_conv_list[:-1]:
                x = F.dropout(F.elu(gcn_conv(x, graph_index, edge_type)))
            node_emb = self.gcn_conv_list[-1](x, graph_index, edge_type)
        elif self.use_conv == 'gcn' or self.use_conv == 'gat':
            for gcn_conv in self.gcn_conv_list[:-1]:
                x = F.dropout(F.elu(gcn_conv(x, graph_index)))
            node_emb = self.gcn_conv_list[-1](x, graph_index)
        elif self.use_conv == 'gps' or self.use_conv == 'anti':
            for gcn_conv in self.gcn_conv_list[:-1]:
                x = F.dropout(gcn_conv(x, graph_index))
            node_emb = self.gcn_conv_list[-1](x, graph_index)
        else:
            raise ValueError("Invalid conv argument.")

        # update the node embedding with the attention mechanism
        atten_node_emb = self.nodes_filter_layer(node_emb, temp_x)
        graph_emb = add_p(atten_node_emb, x_batch).squeeze(dim=1)

        return graph_emb

    def forward(self, left_x, left_graph_index, left_x_batch, right_x=None, right_graph_index=None, right_x_batch=None, label=None,
                left_edge_type=None, right_edge_type=None):

        all_embedding = torch.cat(
            [self.diag_embedding(), self.proce_embedding(), self.atc_embedding()], dim=0)

        # If only one EHR are provided
        if right_x is None:
            return self.encode(all_embedding, left_x, left_graph_index, left_x_batch, left_edge_type)

        # Encode the two ehr graphs
        left_self_global_emb, right_self_global_emb = self.encode(all_embedding, left_x, left_graph_index, left_x_batch, left_edge_type), \
            self.encode(all_embedding, right_x, right_graph_index,
                        right_x_batch, right_edge_type)
            
        # Compute similarity between left and right embeddings
        pair_sim = self.neural_sim_layer(
            left_self_global_emb, right_self_global_emb)
        outputs = self.classifier(pair_sim)
        loss_fct = nn.CrossEntropyLoss()
        sim_loss = loss_fct(outputs.view(-1, 2), label.view(-1))

        if label is not None:
            outputs = [sim_loss, outputs]
            
        return outputs


class OntologyEmbedding(nn.Module):
    """
    generating embeddings based on ontology structure
    """
    def __init__(self, voc, build_tree_func,
                 in_channels=100, out_channels=100, heads=1):
        """
        
        Args:
        voc: Vocabulary object.
        build_tree_func: Function to build the ontology tree.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        heads (int): Number of attention heads.
        
        """
        super(OntologyEmbedding, self).__init__()

        # initial tree edges
        res, graph_voc = build_tree_func(list(voc.idx2word.values()))
        stage_one_edges = build_stage_one_edges(res, graph_voc)
        stage_two_edges = build_stage_two_edges(res, graph_voc)

        self.edges1 = torch.tensor(stage_one_edges)
        self.edges2 = torch.tensor(stage_two_edges)
        self.graph_voc = graph_voc

        # construct model
        assert in_channels == heads * out_channels
        self.g1 = GATConv(in_channels=in_channels,
                          out_channels=in_channels,
                          heads=heads)
        self.g2 = GATConv(in_channels=in_channels,
                          out_channels=out_channels,
                          heads=heads)
        # tree embedding
        num_nodes = len(graph_voc.word2idx)
        self.embedding = nn.Parameter(torch.randn(num_nodes, in_channels))

        # idx mapping: FROM leaf node in graphvoc TO voc
        self.idx_mapping = [self.graph_voc.word2idx[word[2:]]
                            for word in voc.idx2word.values()]

        self.init_params()

    def get_all_graph_emb(self):
        emb = self.embedding
        emb = self.g2(self.g1(emb, self.edges1.to(emb.device)),
                      self.edges2.to(emb.device))
        return emb

    def forward(self):
        emb = self.embedding
        emb = self.g2(self.g1(emb, self.edges1.to(emb.device)),
                      self.edges2.to(emb.device))
        return emb[self.idx_mapping]

    def init_params(self):
        glorot(self.embedding)

class NeuralTensorModule(torch.nn.Module):
    def __init__(self,dim,tensors):
        super(NeuralTensorModule, self).__init__()
        self.dim = dim
        self.tensors = tensors
        self.init_parameters()
        
    def init_parameters(self):
        self.weight_matrix = nn.Parameter(torch.Tensor(self.dim,self.dim,self.tensors))
        self.weight_matrix_block = nn.Parameter(torch.Tensor(2*self.dim,self.tensors))
        self.bias = nn.Parameter(torch.zeros(self.tensors))
        
        glorot(self.weight_matrix)
        glorot(self.weight_matrix_block)
        
    def forward(self,emb_1,emb_2):
        #batch*(dim*tensors)->batch*tensors*dim
        scores = torch.mm(emb_1, self.weight_matrix.view(self.dim, -1)).view(-1, self.dim, self.tensors)
        scores = torch.mm(emb_1,self.weight_matrix.view(self.dim,-1)).view(-1,self.dim,self.tensors).transpose(1,2)
        #batch*tensors
        matrix_scores = torch.bmm(scores,emb_2.unsqueeze(dim=-1)).squeeze(dim=-1)
        
        mat_blo_scores = torch.mm(torch.cat([emb_1,emb_2],dim=1),self.weight_matrix_block)#batch*tensors
        
        combined_scores = nn.functional.tanh(matrix_scores+mat_blo_scores+self.bias)
        return combined_scores
        
                  
class NodeFilterModule(torch.nn.Module):
    def __init__(self,dim):
        super(NodeFilterModule, self).__init__()
        self.dim = dim
        self.weight_matrix = nn.Parameter(torch.rand(1,self.dim))
                
        
    def forward(self,emb_1,emb_2):
        weighted_emb_1 = torch.mul(emb_1,self.weight_matrix).unsqueeze(dim=1)
        scores = torch.sigmoid(torch.bmm(weighted_emb_1,emb_2.unsqueeze(dim=1).transpose(1,2)).squeeze(dim=-1))
        attented_emb = torch.mul(scores.repeat([1,emb_1.size(1)]), emb_1+emb_2)
        return attented_emb