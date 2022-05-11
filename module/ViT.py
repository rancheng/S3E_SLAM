"""
original from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
modified by Ran Cheng <ran.cheng2@mail.mcgill.ca>
update log:
 [+] relational position encoding
 [+] add the hierarchical transformer patch
 [+] add pre-conv encoding
"""
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class tFViT_KL(nn.Module):
    def __init__(self, num_patches, patch_size, pos_dim, emb_dim, depth, heads, mlp_dim, code_dim, channels,
                 pool='mean', dim_head=64, dropout=0., emb_dropout=1e-4):
        super().__init__()
        self.patch_total_size = num_patches * patch_size * patch_size * channels
        self.fvit1 = FViT(num_patches=num_patches,
                          patch_size=patch_size,
                          pos_dim=pos_dim,
                          emb_dim=emb_dim,
                          code_dim=code_dim,
                          depth=depth,
                          heads=heads,
                          mlp_dim=mlp_dim,
                          pool=pool,
                          channels=channels,  # rgbd
                          dim_head=dim_head,
                          dropout=dropout,
                          emb_dropout=emb_dropout)
        self.fvit2 = FViT(num_patches=num_patches,
                          patch_size=patch_size,
                          pos_dim=pos_dim,
                          emb_dim=emb_dim,
                          code_dim=code_dim,
                          depth=depth,
                          heads=heads,
                          mlp_dim=mlp_dim,
                          pool=pool,
                          channels=channels,  # rgbd
                          dim_head=dim_head,
                          dropout=dropout,
                          emb_dropout=emb_dropout)
        self.mlp_head = nn.Sequential(nn.Linear(code_dim * 2, code_dim),
                                      nn.Linear(code_dim, 1))

    def forward(self, patch_img1, pos_data1, patch_img2, pos_data2):
        x1 = self.fvit1(patch_img1, pos_data1)
        x2 = self.fvit2(patch_img2, pos_data2)
        div_out = F.kl_div(x1, x2, reduction='batchmean', log_target=False)
        div_out = div_out / self.patch_total_size  # normalize the KL divergence
        out = 1.0 - torch.sigmoid(div_out)  # flip
        return out


class SharedFViT(nn.Module):
    def __init__(self, num_patches, patch_size, pos_dim, emb_dim, depth, heads, mlp_dim, code_dim, channels,
                 pool='mean', dim_head=64, dropout=0., emb_dropout=1e-4):
        super().__init__()
        self.fvit1 = FViT(num_patches=num_patches,
                          patch_size=patch_size,
                          pos_dim=pos_dim,
                          emb_dim=emb_dim,
                          code_dim=code_dim,
                          depth=depth,
                          heads=heads,
                          mlp_dim=mlp_dim,
                          pool=pool,
                          channels=channels,  # rgbd
                          dim_head=dim_head,
                          dropout=dropout,
                          emb_dropout=emb_dropout)
        # self.fvit2 = FViT(num_patches=num_patches,
        #                   patch_size=patch_size,
        #                   pos_dim=pos_dim,
        #                   emb_dim=emb_dim,
        #                   code_dim=code_dim,
        #                   depth=depth,
        #                   heads=heads,
        #                   mlp_dim=mlp_dim,
        #                   pool=pool,
        #                   channels=channels,  # rgbd
        #                   dim_head=dim_head,
        #                   dropout=dropout,
        #                   emb_dropout=emb_dropout)
        self.mlp_head = nn.Sequential(nn.Linear(code_dim * 2, code_dim),
                                      nn.Linear(code_dim, 1))

    def forward(self, patch_img1, pos_data1, patch_img2, pos_data2):
        x1 = self.fvit1(patch_img1, pos_data1)
        x2 = self.fvit1(patch_img2, pos_data2)
        x = torch.cat((x1, x2), dim=1)
        x = torch.sigmoid(self.mlp_head(x))
        return x

class TwoFViT(nn.Module):
    def __init__(self, num_patches, patch_size, pos_dim, emb_dim, depth, heads, mlp_dim, code_dim, channels,
                 pool='mean', dim_head=64, dropout=0., emb_dropout=1e-4):
        super().__init__()
        self.fvit1 = FViT(num_patches=num_patches,
                          patch_size=patch_size,
                          pos_dim=pos_dim,
                          emb_dim=emb_dim,
                          code_dim=code_dim,
                          depth=depth,
                          heads=heads,
                          mlp_dim=mlp_dim,
                          pool=pool,
                          channels=channels,  # rgbd
                          dim_head=dim_head,
                          dropout=dropout,
                          emb_dropout=emb_dropout)
        self.fvit2 = FViT(num_patches=num_patches,
                          patch_size=patch_size,
                          pos_dim=pos_dim,
                          emb_dim=emb_dim,
                          code_dim=code_dim,
                          depth=depth,
                          heads=heads,
                          mlp_dim=mlp_dim,
                          pool=pool,
                          channels=channels,  # rgbd
                          dim_head=dim_head,
                          dropout=dropout,
                          emb_dropout=emb_dropout)
        self.mlp_head = nn.Sequential(nn.Linear(code_dim * 2, code_dim),
                                      nn.Linear(code_dim, 1))

    def forward(self, patch_img1, pos_data1, patch_img2, pos_data2):
        x1 = self.fvit1(patch_img1, pos_data1)
        x2 = self.fvit2(patch_img2, pos_data2)
        x = torch.cat((x1, x2), dim=1)
        x = torch.sigmoid(self.mlp_head(x))
        return x


# feature-ViT, the patch of ViT is based on the feature points
class FViT(nn.Module):
    def __init__(self, *, num_patches, patch_size, pos_dim, emb_dim, depth, heads, mlp_dim, code_dim, channels,
                 pool='mean', dim_head=64, dropout=0., emb_dropout=1e-4):
        super().__init__()
        self.num_patches = num_patches  # 128
        self.patch_dim = channels * patch_size * patch_size
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.patch_embedding = nn.Sequential(
            Rearrange('b p h w c -> b p (h w c)'),
            nn.Linear(self.patch_dim, emb_dim)
        )
        self.pos_embedding = nn.Linear(pos_dim, emb_dim)
        self.cls_pos_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(emb_dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        # regression
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, code_dim)
        )

    def forward(self, patch_img, pos_data):
        x = self.patch_embedding(patch_img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        cls_pos_tokens = repeat(self.cls_pos_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_emb = self.pos_embedding(pos_data)
        pos_emb = torch.cat((cls_pos_tokens, pos_emb), dim=1)
        x += pos_emb[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
