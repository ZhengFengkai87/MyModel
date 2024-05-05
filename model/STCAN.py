import torch.nn as nn
import torch
from torchinfo import summary


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()

        self.k = kernel_size[1]
        self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        s = int((self.k - 1) / 2)
        x = nn.functional.pad(x, (s, s, 0, 0), 'replicate')
        x = self.avg(x)

        return x


# 自适应邻接矩阵掩码
class DGP(nn.Module):
    def __init__(self):
        super(DGP, self).__init__()

    def forward(self, attn_score, adp):
        mask = adp
        attn_score = attn_score * mask

        return attn_score


class SpatialAttention(nn.Module):
    def __init__(self, model_dim, num_heads, mask=False):
        super(SpatialAttention, self).__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        
        self.dgp = DGP()
        

    def forward(self, query, key, value, adp):
        # Q, K, V (batch_size, T, N, model_dim)
        N = query.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, T, N, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, T, head_dim, N)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, T, N, N)

        if self.mask:
            mask = torch.ones(
                N, N, dtype=torch.bool, device=query.device
            )
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place
        
        attn_score = self.dgp(attn_score, adp)

        attn_score = torch.softmax(attn_score, dim=-1)
        
        # mask to top 30%
        topk_values, topk_indices = torch.topk(attn_score, k=int(attn_score.shape[-1]*0.3), dim=-1)
        mask = torch.zeros_like(attn_score)
        mask.scatter_(-1, topk_indices, 1)
        attn_score = attn_score * mask

        # (num_heads * batch_size, T, N, N)
        # (num_heads * batch_size, T, N, head_dim)
        return attn_score, value


class TemporalAttention(nn.Module):
    def __init__(self, model_dim, num_heads, mask=False):
        super(TemporalAttention, self).__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # Q, K, V (batch_size, N, T, model_dim)
        T = query.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, N, T, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, N, head_dim, T)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, N, T, T)

        if self.mask:
            mask = torch.ones(
                T, T, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)

        # (num_heads * batch_size, N, T, T)
        # (num_heads * batch_size, N, T, head_dim)
        return attn_score, value


class STAttnFusion(nn.Module):
    def __init__(self, model_dim):
        super(STAttnFusion, self).__init__()

        self.weighting_sum_conv = nn.Conv3d(in_channels=model_dim, out_channels=model_dim, kernel_size=(2,1,1))
        self.fs = nn.Linear(model_dim, model_dim)
        self.ft = nn.Linear(model_dim, model_dim)
    
    def forward(self, x, y):
        # (B, D, 1, N, T)   (N, C, D, H, W)
        x1 = self.fs(x).transpose(1, 3).unsqueeze(2)
        y1 = self.ft(y).transpose(1, 3).unsqueeze(2)
        weighting_sum = self.weighting_sum_conv(torch.cat([x1, y1], dim=2))
        weighting_sum = weighting_sum.squeeze(2).transpose(1, 3)

        z = torch.sigmoid(weighting_sum)
        out = torch.add(torch.mul(z, x), torch.mul(1 - z, y))
        
        return out


class STCrossAttentionBlock(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0):
        super(STCrossAttentionBlock, self).__init__()

        self.spatial_conv = nn.Conv2d(in_channels=model_dim, out_channels=model_dim, kernel_size=(1,1), bias=True)
        # self.temporal_conv = nn.Conv2d(in_channels=model_dim, out_channels=model_dim, kernel_size=(1,1), bias=True)
        self.temporal_conv = moving_avg(kernel_size=(1,5), stride=1)

        self.SAttn = SpatialAttention(model_dim, num_heads, False)
        self.TAttn = TemporalAttention(model_dim, num_heads, False)

        self.s_proj = nn.Linear(model_dim, model_dim)
        self.t_proj = nn.Linear(model_dim, model_dim)
        self.s_ln1 = nn.LayerNorm(model_dim)
        self.t_ln1 = nn.LayerNorm(model_dim)
        self.s_ln2 = nn.LayerNorm(model_dim)
        self.t_ln2 = nn.LayerNorm(model_dim)
        self.s_dropout1 = nn.Dropout(dropout)
        self.t_dropout1 = nn.Dropout(dropout)
        self.s_dropout2 = nn.Dropout(dropout)
        self.t_dropout2 = nn.Dropout(dropout)
        self.s_feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.t_feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )

        self.fusion = STAttnFusion(model_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(model_dim)

    def forward(self, x, adp):  # x:(B, T, N, D)
        batch_size = x.shape[0]
        s_input = self.spatial_conv(x.transpose(1,3)).transpose(1,3)  # (B, T, N, D)
        t_input = self.temporal_conv(x.transpose(1,3)).transpose(1,3)  # (B, T, N, D)
        
        # s_score (num_heads * batch_size, T, N, N)
        # s_v (num_heads * batch_size, T, N, head_dim)
        s_score, s_v = self.SAttn(s_input, s_input, s_input, adp)

        # t_score (num_heads * batch_size, N, T, T)
        # t_v (num_heads * batch_size, N, T, head_dim)
        t_score, t_v = self.TAttn(t_input, t_input, t_input)

        # s_out (num_heads * batch_size, T, N, head_dim)
        # t_out (num_heads * batch_size, N, T, head_dim)
        s_out = s_score @ t_v.transpose(1, 2)
        t_out = t_score @ s_v.transpose(1, 2)

        # s_out (batch_size, T, N, head_dim * num_heads = model_dim)
        # t_out (batch_size, N, T, head_dim * num_heads = model_dim)
        s_out = torch.cat(torch.split(s_out, batch_size, dim=0), dim=-1)
        t_out = torch.cat(torch.split(t_out, batch_size, dim=0), dim=-1)
        
        # t_out (batch_size, T, N, head_dim * num_heads = model_dim)
        t_out = t_out.transpose(1, 2)

        s_out = self.s_ln1(self.s_dropout1(self.s_proj(s_out)) + x)
        t_out = self.t_ln1(self.t_dropout1(self.t_proj(t_out)) + x)

        s_residual = s_out
        s_out = self.s_ln2(self.s_dropout2(self.s_feed_forward(s_out)) + s_residual)
        t_residual = t_out
        t_out = self.t_ln2(self.t_dropout2(self.t_feed_forward(t_out)) + t_residual)

        # out (batch_size, T, N, model_dim)
        out = self.fusion(s_out, t_out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout(out)
        out = self.ln(residual + out)

        return out


class STCAN(nn.Module):
    def __init__(
        self,
        device,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
    ):
        super(STCAN, self).__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.cross_attn_blocks = nn.ModuleList(
            [
                STCrossAttentionBlock(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        adp = torch.softmax(torch.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        # (batch_size, in_steps, num_nodes, model_dim)
        for block in self.cross_attn_blocks:
            x = block(x, adp)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).reshape(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out


if __name__ == "__main__":
    model = STCAN(207, 12, 12)
    summary(model, [64, 12, 207, 3])
