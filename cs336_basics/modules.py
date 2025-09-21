from math import sqrt
import torch
import torch.nn as nn
from einops import rearrange, repeat,einsum

#记住初始化参数
class Linear(nn.Module):
    def __init__(self, input_dim, output_dim,device=None, dtype=None):
        super(Linear, self).__init__()  
        self.linear = nn.Parameter(torch.randn(input_dim, output_dim,device=device,dtype=dtype))
        self.in_feature = input_dim
        self.out_feature = output_dim
    
    def _init_weights(self):
        std = (2 / (self.in_feature + self.out_feature)) ** 0.5
        nn.init.trunc_normal_(self.linear, std=std,a = -3*std,b = 3*std)

    def forward(self, x):
        return einsum(x,self.linear,'... input,input output -> ... output')

#torch[]索引可以直接索引多个维度
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.d_dim = embedding_dim
        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim,device=device,dtype=dtype))

    def _init_weights(self):
        nn.init.trunc_normal_(self.embedding,a = -3,b = 3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # batch,seq = token_ids.shape
        # token_ids_plat = rearrange(token_ids,'b seq->(b seq)')
        # embedding = self.embedding[token_ids_plat]
        # embedding = rearrange(embedding,'(b h) e->b h e',b=batch)

        return self.embedding[token_ids]
    
#在prenorm应该使用float32计算来增加精度
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(dim,device=device,dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        sqrt_mean_square = torch.sqrt(torch.mean(x*x,dim=-1,keepdim=True)+self.eps)
        result = x * self.gain / sqrt_mean_square

        return result.to(in_dtype)

def SwiGLu(x):
    return x * torch.sigmoid(x)

class SwiGLu_FFN(nn.Module):
    def __init__(self, dim,dff,device=None,dtype=None):
        super().__init__()
        self.dim = dim
        self.dff = dff
        self.weights_swi = Linear(dim,dff,device,dtype)
        self.weights_glu = Linear(dim,dff,device,dtype)
        self.output = Linear(dff,dim,device,dtype)
    def forward(self,x):
        swish = SwiGLu(self.weights_swi(x))
        glu = self.weights_glu(x)
        result = swish * glu
        return self.output(result)


#常用索引rearrange [...,1::2]取偶数位,torch.stack torch.cat
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k // 2
        theta_k = 1.0 / theta ** (torch.arange(0,self.d_k).to(torch.float32)/ self.d_k)
        index_k = torch.linspace(0,max_seq_len-1,steps=max_seq_len)
        theta_matrix = einsum(index_k,theta_k,"index,theta -> index theta")
        self.register_buffer("theta_matrix_sin",torch.sin(theta_matrix),persistent=False)
        self.register_buffer("theta_matrix_cos",torch.cos(theta_matrix),persistent=False)

    
    def forward(self, x: torch.Tensor, token_positions=None) -> torch.Tensor:
        if token_positions is None:
            b,seq,_ = x.shape
            token_positions = repeat(torch.arange(0,seq,step = 1),"seq-> b seq")

        sin_matrix = repeat(self.theta_matrix_sin,"b h -> b (h h1)",h1 = 2)
        cos_matrix = repeat(self.theta_matrix_cos,"b h -> b (h h1)",h1 = 2)
        x_sin = einsum(x,sin_matrix[token_positions],"... seq d,... seq d -> ... seq d")
        x_cos = einsum(x,cos_matrix[token_positions],"... seq d,... seq d -> ... seq d")
        x_sin = rearrange(x_sin,"... seq (d1 d2) -> ... seq d1 d2",d2 = 2)
        x_cos = rearrange(x_cos,"... seq (d1 d2) -> ... seq d1 d2",d2 = 2)
        x1,x2 = x_sin.unbind(-1)
        result = torch.stack((-x2,x1),dim=-1) + x_cos
        return rearrange(result,"... d d1 -> ... (d d1)")

def Softmax(x:torch.Tensor,dim: int) -> torch.Tensor:
    
    max_num = torch.max(x,dim=dim,keepdim=True).values
    #print(max_num.shape)
    x = x - max_num
    x = torch.exp(x)
    return x / torch.sum(x,dim = dim,keepdim=True)
#torch.where torch.triu
def Scaled_dot_product_Attention(keys,queries,values,mask=None) -> torch.Tensor:
    """
    keys:b ... seq d_k q:b ... seq d_k 
    mask:seq seq
    """
    d_k = keys.shape[-1]
    qk = einsum(queries,keys,"... seq d_k,... seq1 d_k -> ... seq seq1") / sqrt(d_k)
    mask_matrix = torch.where(mask,qk,-torch.inf) if mask is not None else qk
    qk = Softmax(mask_matrix,dim=-1)
    return einsum(qk,values,"... seq seq1,... seq1 d_v -> ... seq d_v")

class Multihead_Self_Attention(nn.Module):
    def __init__(self, d_model, num_heads,max_seq=1024):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.queries = nn.Parameter(torch.ones([d_model,d_model]))
        self.keys = nn.Parameter(torch.ones([d_model,d_model]))
        self.values = nn.Parameter(torch.ones([d_model,d_model]))
        self.output = nn.Parameter(torch.randn([d_model,d_model]))
        self.register_buffer("masked",torch.tril(torch.ones(max_seq,max_seq).to(torch.bool)),persistent=False)
    
    def forward(self,x):
        b,seq,d = x.shape
        queries = x @ self.queries.T
        keys = x @ self.keys.T
        values = x @ self.values.T
        queries = rearrange(queries,"... seq (num h) -> ... num seq h",num = self.num_heads)
        keys = rearrange(keys,"... seq (num h) -> ... num seq h",num = self.num_heads)
        values = rearrange(values,"... seq (num h) -> ... num seq h",num = self.num_heads)
        result = Scaled_dot_product_Attention(keys,queries,values,self.masked[:seq,:seq])
        return rearrange(result,"b num seq h -> b seq (num h)") @ self.output.T
    
class Multihead_Self_Attention_with_RoPE(nn.Module):
    def __init__(self, d_model: int,num_heads: int,max_seq_len: int,theta):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.queries = nn.Parameter(torch.ones([d_model,d_model]))
        self.keys = nn.Parameter(torch.ones([d_model,d_model]))
        self.values = nn.Parameter(torch.ones([d_model,d_model]))
        self.output = nn.Parameter(torch.randn([d_model,d_model]))
        self.rope = RoPE(theta,self.d_head,max_seq_len)
        self.register_buffer("masked",torch.tril(torch.ones(max_seq_len,max_seq_len).to(torch.bool)),persistent=False)
    
    def forward(self,x,token_positions=None):
        b,seq,d = x.shape
        queries = x @ self.queries.T
        keys = x @ self.keys.T
        values = x @ self.values.T
        queries = rearrange(queries,"... seq (num h) -> ... num seq h",num = self.num_heads)
        keys = rearrange(keys,"... seq (num h) -> ... num seq h",num = self.num_heads)
        queries = self.rope(queries,token_positions)
        keys = self.rope(keys,token_positions)
        values = rearrange(values,"... seq (num h) -> ... num seq h",num = self.num_heads)
        result = Scaled_dot_product_Attention(keys,queries,values,self.masked[:seq,:seq])
        return rearrange(result,"b num seq h -> b seq (num h)") @ self.output.T


if __name__ == "__main__":
    K = torch.randn([3,2,3])
    Q = torch.randn([3,2,3])
    V = torch.randn([10,10])
    print(V[:5,:5])
