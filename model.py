import config

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPLayer, self).__init__()
        self.mlp = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        output = F.relu(self.mlp(input))
        return output


class NeuralCF(nn.Module):
    def __init__(self):
        super(NeuralCF, self).__init__()
        self.embedding_user_mf = nn.Embedding(config.N_USER + 1, config.D_EMBEDDING)
        self.embedding_item_mf = nn.Embedding(config.N_ITEM + 1, config.D_EMBEDDING)
        self.embedding_user_mlp = nn.Embedding(config.N_USER + 1, config.D_EMBEDDING)
        self.embedding_item_mlp = nn.Embedding(config.N_ITEM + 1, config.D_EMBEDDING)

        self.mlp_layers = nn.ModuleList( 
            [ MLPLayer(config.D_EMBEDDING * 2, config.LAYERS[0]) ]
            +
            [ MLPLayer(config.LAYERS[i], config.LAYERS[i+1]) for i in range(len(config.LAYERS)-1)]
        )
        self.affine_output = nn.Linear(config.LAYERS[-1] + config.D_EMBEDDING, 1)

    def forward(self, user_id, item_id):
        embedded_user_mf = self.embedding_user_mf(user_id)   # B x embedding
        embedded_item_mf = self.embedding_item_mf(item_id)   # B x embedding
        embedded_user_mlp = self.embedding_user_mlp(user_id)   # B x embedding
        embedded_item_mlp = self.embedding_item_mlp(item_id)   # B x embedding

        gmf_output = torch.mul(embedded_user_mf, embedded_item_mf)
        mlp_output = torch.cat((embedded_user_mlp, embedded_item_mlp), -1) # B X embedding*2

        for mlp in self.mlp_layers:
            mlp_output = mlp(mlp_output) # B x last_layer_dim

        neumf_output = torch.cat((gmf_output, mlp_output), -1) # B x (last_layer_dim + embedding)
        logits = self.affine_output(neumf_output) # B x 1
        rating = torch.sigmoid(logits)

        return rating

class Model():
    def __init__(self):
        self.model = NeuralCF()
        if config.USE_GPU:
            self.model = self.model.cuda()
