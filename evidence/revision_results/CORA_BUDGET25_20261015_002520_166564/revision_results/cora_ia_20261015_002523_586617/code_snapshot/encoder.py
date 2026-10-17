import torch
import torch.nn as nn

from pos_enc import PEG
from utils import get_activation
from conv import GATConv
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class Model(torch.nn.Module):
    def __init__(self, encoder, out_dim, args=None):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(args.embed_dim, out_dim)
    
    def forward(self, x, edge_index, eigvals=None, eigvecs=None, mask_tokens=None, PE=None):
        h, pe = self.encoder.embed(x, edge_index, eigvals, eigvecs, mask_tokens, PE)
        pred = self.classifier(h)
        return pred, pe



class GraphEncoder(torch.nn.Module):
    def __init__(self, out_dim, args=None):
        super().__init__()
        self.gnn_dropout = args.gnn_dropout
        self.use_checkpoint = bool(getattr(args, "use_checkpoint", False))

        self.num_layer = args.enc_gnn_layer
        emb_dim = args.embed_dim
        hid_dim = emb_dim // args.heads

        self.gnns = nn.ModuleList()
        self.activations = nn.ModuleList()

        if self.num_layer == 1:
            self.gnns.append(
                GATConv(
                    in_channels=args.feat_dim,
                    out_channels=out_dim,
                    heads=args.heads,
                    args=args,
                )
            )
        else:
            self.gnns.append(
                GATConv(
                    in_channels=args.feat_dim,
                    out_channels=hid_dim,
                    heads=args.heads,
                    args=args,
                )
            )

            for layer in range(self.num_layer - 1):
                self.gnns.append(
                    GATConv(
                        in_channels=emb_dim,
                        out_channels=hid_dim,
                        heads=args.heads,
                        args=args,
                    )
                )
                self.activations.append(get_activation(args.gnn_activation))

        self.pe_enc = PEG(args=args)

    def _run_gnn_layer(self, layer, h, pe, edge_index):
        if self.use_checkpoint and self.training:
            return checkpoint(
                lambda h_in, pe_in: self.gnns[layer](h_in, pe_in, edge_index),
                h,
                pe,
                use_reentrant=False,
            )
        return self.gnns[layer](h, pe, edge_index)

    def forward(self, x, x_masked, edge_index, PE=None, PE_noise=None):
        pe = self.pe_enc(PE)
        pe_noise = self.pe_enc(PE_noise)

        h = x
        h_masked = x_masked

        for layer in range(self.num_layer):
            h_in = F.dropout(h, p=self.gnn_dropout, training=self.training)
            h, pe_noise = self._run_gnn_layer(layer, h_in, pe_noise, edge_index)

            h_masked_in = F.dropout(h_masked, p=self.gnn_dropout, training=self.training)
            h_masked, pe = self._run_gnn_layer(layer, h_masked_in, pe, edge_index)

            if layer != self.num_layer - 1:
                h = self.activations[layer](h)
                h_masked = self.activations[layer](h_masked)

        return h_masked, pe_noise

    def embed(self, x, edge_index, PE=None):
        pe = self.pe_enc(PE)
        h = x

        for layer in range(self.num_layer):
            h_in = F.dropout(h, p=self.gnn_dropout, training=self.training)
            h, pe = self.gnns[layer](h_in, pe, edge_index)

            if layer != self.num_layer - 1:
                h = self.activations[layer](h)

        return h, pe
