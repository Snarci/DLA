import torch
from torch import nn

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


class MemA(nn.Module):
    def __init__(self,shared_memory, input_size, output_size, dropout=0.4):
        super(MemA, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.shared_memory = shared_memory

        self.x = nn.Linear(self.input_size, self.output_size)
        self.x_out = nn.Linear(self.output_size, self.output_size)

        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(self.output_size)
        self.dropout = nn.Dropout(self.dropout)
    def forward(self, x):
        x = self.x(x)
        ATTENTION_BLOCKS = self.shared_memory.shape[-1]
        for i in range(ATTENTION_BLOCKS):
            x = x + torch.matmul(self.shared_memory[:, :, :, i],x)
        x = self.dropout(self.norm(x))
        x = self.x_out(x)
        return x 
       
class Block(nn.Module):
    def __init__(self, token_size=512):
        super(Block, self).__init__()

        self.token_size = token_size
        expansion = 2
        self.feed_forward = nn.Sequential(
            nn.Linear(self.token_size, self.token_size*expansion),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.token_size*expansion, self.token_size),
        )
        self.layer_norm = nn.LayerNorm(self.token_size)
        self.attention = nn.MultiheadAttention(self.token_size, 8)
        self.attention_layer_norm = nn.LayerNorm(self.token_size)
        

        
    def forward(self, x):
        x = self.attention_layer_norm(self.attention(x, x, x)[0] + x)
        x = self.layer_norm(self.feed_forward(x) + x)
        
        return x
class TransformerM(nn.Module):
    def __init__(self,depth=8,memory_blocks=16,emb=32, token_size=512):
        super(TransformerM, self).__init__()
        self.depth = depth
        self.memory_blocks = memory_blocks
        self.emb = emb
        self.token_size = token_size
        self.memory = nn.Parameter(torch.rand(1, self.emb, self.emb,self.memory_blocks))
        self.blocks = nn.ModuleList([Block(self.token_size) for _ in range(self.depth)])
        self.mema = MemA(self.memory, self.token_size, self.token_size)
    def forward(self, x):
        for block in self.blocks:
            x = self.mema(x) + x
            x = block(x)
        return x

class ViTSM(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,mem_blocks, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerM(depth=8,memory_blocks=mem_blocks,emb= (num_patches + 1), token_size=dim)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
if __name__ == '__main__':
    model = ViTSM(
    mem_blocks=128,
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)
    print(model)
    x = torch.randn(1, 3, 256, 256)
    print(model(x).shape)
    #try a backward pass
    y = model(x)
    y.sum().backward()
    print('done')