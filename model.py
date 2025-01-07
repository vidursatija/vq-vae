from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualModule(nn.Module):
    def __init__(self, hidden_size: int):
        super(ResidualModule, self).__init__()
        self.rms_norm1 = nn.RMSNorm((hidden_size, 8, 8))
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.rms_norm2 = nn.RMSNorm((hidden_size, 8, 8))
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 1, padding=0)
    
    def forward(self, x):
        inp = x
        x = self.rms_norm1(x)
        x = self.conv1(x)
        x = F.relu(x)

        x = inp + x
        inp = x

        x = self.rms_norm2(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = inp + x
        return x

class Encoder(nn.Module):
    def __init__(self, hidden_size: int):
        super(Encoder, self).__init__()
        self.strided_conv1 = nn.Conv2d(3, hidden_size, 4, stride=2, padding=1)
        self.strided_conv2 = nn.Conv2d(hidden_size, hidden_size, 4, stride=2, padding=1)
        self.residual_module1 = ResidualModule(hidden_size)
        self.residual_module2 = ResidualModule(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.strided_conv1(x)
        x = F.relu(x)

        x = self.strided_conv2(x)
        x = F.relu(x)

        x = self.residual_module1(x)
        x = self.residual_module2(x)

        return x

class Decoder(nn.Module):
    def __init__(self, hidden_size: int):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(hidden_size, hidden_size, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_size, 3, 4, stride=2, padding=1)
        self.residual_module1 = ResidualModule(hidden_size)
        self.residual_module2 = ResidualModule(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_module1(x)
        x = self.residual_module2(x)

        x = self.deconv1(x)
        x = F.relu(x)

        x = self.deconv2(x)

        return x

class QuantizedDict(nn.Module):
    def __init__(self, hidden_size: int, num_embeddings: int):
        super(QuantizedDict, self).__init__()
        self.embedding = nn.Parameter(torch.randn(num_embeddings, hidden_size))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x is of shape (batch_size, hidden_size, height, width)
        # We want to find the closest embedding for each pixel in x
        # lets make this (batch_size * height * width, hidden_size)
        b, hs, h, w = x.shape
        inp = x

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, x.shape[-1])

        # Now we want to find the closest embedding for each pixel
        l2_distances = torch.cdist(x, self.embedding)  # (batch_size * height * width, num_embeddings)
        closest_embeddings_indx = torch.argmin(l2_distances, dim=1)  # (batch_size * height * width)
        closest_embeddings = self.embedding[closest_embeddings_indx]  # (batch_size * height * width, hidden_size)

        # Reshape back to (batch_size, height, width, hidden_size)
        closest_embeddings = closest_embeddings.reshape(b, h, w, hs).permute(0, 3, 1, 2)

        # now we need to pass the gradients back to the embeddings so we use a trick
        oup = inp + (closest_embeddings - inp).detach()
        # this makes the gradients flow through the input and not the embeddings

        # now we calculate 2 losses for the encoder + dictionary update -> oup - sg[closest_embeddings], sg[oup] - closest_embeddings
        dictionary_loss = F.mse_loss(inp.detach(), closest_embeddings)  # this is the loss for the dictionary so that we can update the embeddings
        commitment_loss = F.mse_loss(inp, closest_embeddings.detach())  # this is the loss for the encoder so that the encoder learns

        return oup, dictionary_loss, commitment_loss

class VQVAE(nn.Module):
    def __init__(self, hidden_size: int, num_embeddings: int, commitment_beta: float = 1.0):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(hidden_size)
        self.quantized_dict = QuantizedDict(hidden_size, num_embeddings)
        self.decoder = Decoder(hidden_size)
        self.commitment_beta = commitment_beta
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        enc_emb = self.encoder(x)
        dec_emb, dictionary_loss, commitment_loss = self.quantized_dict(enc_emb)
        decoded = self.decoder(dec_emb)

        reconst_loss = F.mse_loss(decoded, x)

        total_loss = reconst_loss + dictionary_loss + self.commitment_beta * commitment_loss

        ret_dict = {
            "loss": total_loss,
            "reconst_loss": reconst_loss,
            "dictionary_loss": dictionary_loss,
            "commitment_loss": commitment_loss,
            "decoded": decoded
        }

        return ret_dict