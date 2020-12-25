import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        # xの値をLinearに通して、それぞれqkvにする
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # multi head attentionに変更　hがヘッド数, nはパッチ数
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv) # q, k, v: batch x head数 x (patch数 + 1) x headごとのdimention

        # パッチ間の内積を取ることで、queryとkeyの相関関係を算出する。 ここでのiとjは、(パッチ+1)を表す。
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale # dots: batch x head数 x (patch数 + 1) x (patch数 + 1)

        # マスクをする際のマスクの値を算出
        # finfoによって、浮動小数点の情報が得られる。 torch.finfo(dots.dtype).maxは浮動小数点の最大値
        mask_value = -torch.finfo(dots.dtype).max # mask_valueは事実上のマイナス無限大を表す

        if mask is not None: # mask: 1 x (パッチ数h) x (パッチ数h)
            # マスクをパッチ数 + 1にしてxと次元を合わせる (cls_tokenの分を追加)
            mask = F.pad(mask.flatten(1), (1, 0), value = True)  # mask: 1 x (パッチ数hw + 1)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'

            # マスク行列同士の論理積をとる。 Noneを入れることで次元を1つ増やす (unsqueezeと同じ効果)
            mask = mask[:, None, :] * mask[:, :, None] # mask: 1 x (パッチ数hw + 1) x (パッチ数hw + 1)

            # maskの値がFalseの部分に、マイナス無限大を設定する。マイナス無限大の部分は後続のsoftmaxを介すとゼロになり、valueをゼロにできる
            dots.masked_fill_(~mask, mask_value)
            del mask
        # ソフトマックスを通すことで 0 ~ 1の範囲に収める
        # ここでマスクを掛けられた部分はゼロになる
        attn = dots.softmax(dim=-1)

        # queryとkeyの相関関係とvalueの内積を取ることで、各queryの重要度ベクトル？的なものが計算できる
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)') # out: batch x (patch数 + 1) x input_dimension (xと同じ次元に戻る)
        # 全結合層に通す
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = patch_size
        # 位置エンコーディングは学習可能なパラメータとする
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        # クラスを表す、クラストークン、こちらも学習可能
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # self.cls_token: 1 x 1 x (embedding_dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, mask = None):
        # pは画像における各パッチの長さ(高さ、幅で共通)
        p = self.patch_size

        # (h w): パッチの数, (p1 p2 c): 各パッチの大きさ
        # (B, C, H, W)を、(B, 合計Patch数, 各Patchの面積) に変更
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p) # x: (batch) x (patch数) x (patch面積)

        # Linear層を通すことで、埋め込み空間の特徴ベクトルに変更
        x = self.patch_to_embedding(x) # x: (batch) x (patch数) x (embedding_dim)
        b, n, _ = x.shape  # b: batch, n: patch数

        # cls_tokenをBatch数だけ増やす
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) # cls_tokens: (batch) x 1 x (embedding_dim)
        # class tokenを元のベクトルに結合する
        x = torch.cat((cls_tokens, x), dim=1) # x: (batch) x (patch数 + 1) x (embedding_dim)

        # 位置エンコーディングを足し合わせる
        x += self.pos_embedding[:, :(n + 1)] # x: (batch) x (patch数 + 1) x (embedding_dim)
        x = self.dropout(x)

        # transformerの処理
        x = self.transformer(x, mask) # x: (batch) x (patch数 + 1) x (dimension)

        # (パッチ数 + class token=1)の次元に沿って平均を取る
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0] # x: (batch) x (dimension)

        x = self.to_latent(x)
        return self.mlp_head(x)  # x: (batch) x (class数)
