import torch
from torch import nn
from transformers import BertModel

class Encoder(nn.Module):
    def __init__(self, model_name, device):
        super(Encoder, self).__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        self.device = torch.device(device)

    def forward(self, encode):
        encode_output = self.encoder(**encode)
        encode_output = encode_output.last_hidden_state[:, 0, :]

        return encode_output


class CausalEncoder(nn.Module):
    def __init__(self, cause_encoder, effect_encoder, semantic_encoder, device):
          super(CausalEncoder, self).__init__()
          self.device = torch.device(device)
          self.cause_encoder = cause_encoder
          self.effect_encoder = effect_encoder
          self.semantic_encoder = semantic_encoder
          for param in self.semantic_encoder.parameters():
            param.requires_grad = False


    def forward(self, cause_encode, effect_encode):
        cause_encoding = self.cause_encoder(cause_encode)
        effect_encoding = self.effect_encoder(effect_encode)
        with torch.no_grad():
            semantic_cause_encoding = self.semantic_encoder(cause_encode)
            semantic_effect_encoding = self.semantic_encoder(effect_encode)

        return cause_encoding, effect_encoding, semantic_cause_encoding, semantic_effect_encoding

