import torch
import torch.nn as nn


def trunc_normal_(tensor, mean=0., std=1.):
    with torch.no_grad():
        return tensor.normal_(mean=mean, std=std).clamp_(-2*std, 2*std)
    
# 이미지 임베딩해보기
class EmbeddingLayer(nn.Module):
    def __init__(self, in_chans, embed_dim, img_size, patch_size):
        super().__init__()
        # TODO
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        
        self.num_patches = (img_size // patch_size) ** 2

        # 패치를 임베딩하는 Conv2d: 패치 크기를 커널과 스트라이드로 설정
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                              kernel_size=patch_size, 
                              stride=patch_size)

        # 학습 가능한 [CLS] 토큰 생성 (1개 추가)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 학습 가능한 위치 임베딩 (CLS + 패치 수만큼 생성)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)  # 위치 임베딩 초기화


    def forward(self, x):  
        # shape of x : (N, 3, 32, 32)
        # shape of return : (N, #ofPatches+1, embed_dim)

        # 각 patch를 임베딩 후 class token과 concat
        # positional embedding까지 더하기
        # TODO

        N = x.shape[0]

        # Conv2d로 패치 임베딩 수행, shape: (N, embed_dim, num_patches**0.5, num_patches**0.5)
        x = self.proj(x).flatten(2).transpose(1, 2)  # shape: (N, num_patches, embed_dim)

        # [CLS] 토큰을 배치 크기에 맞게 확장
        cls_tokens = self.cls_token.expand(N, -1, -1)  # shape: (N, 1, embed_dim)

        # [CLS] 토큰과 패치 임베딩을 concat
        x = torch.cat((cls_tokens, x), dim=1)  # shape: (N, num_patches + 1, embed_dim)

        # 위치 임베딩 추가
        x = x + self.pos_embed  # shape: (N, num_patches + 1, embed_dim)
        return x




    
    
class MSA(nn.Module):
    def __init__(self, dim=192, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x