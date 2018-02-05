import torch
import torch.nn as nn

#
# For serverless inference, embedding is taken out from Peng's original KimCNN implementation
# (https://github.com/castorini/Castor/tree/master/kim_cnn)
#
class Embedding:
    def __init__(self, config, vectors):
        self.config = config
        if config.mode not in ['rand', 'static', 'non-static', 'multichannel']:
            print("Unsupport Mode: " + config.mode)
            exit()

        self.rand_embedding = nn.Embedding(config.words_num, config.words_dim) # default random embedding
        self.static_embedding = nn.Embedding(config.embed_num, config.embed_dim)
        self.non_static_embedding = nn.Embedding(config.embed_num, config.embed_dim)
        self.static_embedding.weight.data.copy_(vectors)
        self.static_embedding.weight.requires_grad = False
        self.non_static_embedding.weight.data.copy_(vectors)

        if config.cuda:
            self.rand_embedding.cuda(config.gpu)
            self.static_embedding.cuda(config.gpu)
            self.non_static_embedding.cuda(config.gpu)


    def embed(self, batch):
        batch_text = batch.text
        embedded = None
        if self.config.mode == 'rand':
            embedded = self.rand_embedding(batch_text)
            embedded = embedded.unsqueeze(1)
        elif self.config.mode == 'static':
            embedded = self.static_embedding(batch_text)
            embedded = embedded.unsqueeze(1)
        elif self.config.mode == 'non-static':
            embedded = self.non_static_embedding(batch_text)
            embedded = embedded.unsqueeze(1)
        elif self.config.mode == 'multichannel':
            static_embedded = self.static_embedding(batch_text)
            non_static_embedded = self.non_static_embedding(batch_text)
            embedded = torch.stack([non_static_embedded, static_embedded], dim=1)
        else:
            print("Unsupport Mode: " + config.mode)
            exit()
        
        return embedded
