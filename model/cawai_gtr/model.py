import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class Encoder(nn.Module):
    def __init__(self, model_name, device):
        super(Encoder, self).__init__()
        self.encoder = SentenceTransformer(model_name)
        self.encoder.max_seq_length = 256
        self.device = torch.device(device)

    def forward(self, text):
        encode_output = self.encoder(text)
        return encode_output

    def tokenize(self, text):
        return self.encoder.tokenize(text)

class CausalEncoder(nn.Module):
    def __init__(self, cause_encoder, effect_encoder, semantic_encoder, device):
          super(CausalEncoder, self).__init__()
          self.device = torch.device(device)
          self.cause_encoder = cause_encoder.to(self.device)
          self.effect_encoder = effect_encoder.to(self.device)
          self.semantic_encoder = semantic_encoder.to(self.device)
          for param in self.semantic_encoder.parameters():
            param.requires_grad = False


    def forward(self, cause, effect):
        cause_encoding = self.cause_encoder(cause)
        effect_encoding = self.effect_encoder(effect)
        with torch.no_grad():
            semantic_cause_encoding = self.semantic_encoder(cause)
            semantic_effect_encoding = self.semantic_encoder(effect)

        return cause_encoding, effect_encoding, semantic_cause_encoding, semantic_effect_encoding


