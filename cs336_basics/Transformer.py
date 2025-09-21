#from cs336_basics.modules import *
from modules import *
from jaxtyping import Bool, Float, Int
from torch import Tensor

class Transformer_Block(nn.Module):
    def __init__(self, d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,):
        super().__init__()
        self.attention = Multihead_Self_Attention_with_RoPE(d_model,num_heads,max_seq_len,theta)
        self.rms1 = RMSNorm(d_model)
        self.ffn = SwiGLu_FFN(d_model,d_ff)
        self.rms2 = RMSNorm(d_model)

    def forward(self,x):
        b,seq,_ = x.shape
        token_positions = torch.arange(0,seq)
        token_positions = repeat(token_positions,"seq -> b seq",b = b)
        attn = self.attention(self.rms1(x),token_positions)
        x = x + attn
        ffn = self.ffn(self.rms2(x))
        x = x + ffn
        return x

class Transformer_My(nn.Module):
    def __init__(self, vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float):
        super().__init__()
        
        self.transformers = nn.ModuleList([Transformer_Block(d_model,num_heads,d_ff,context_length,rope_theta) for _ in range(num_layers)])
        self.embebbing = Embedding(vocab_size,d_model)

        self.rms = RMSNorm(d_model)
        self.output = Linear(d_model,vocab_size)

    def forward(self,x):
        tokens = self.embebbing(x)
        for transformer in self.transformers:
            tokens = transformer(tokens)
        tokens = self.rms(tokens)
        tokens = self.output(tokens)
        return tokens

#torch.gather torch.logsumexp
def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, ""]):
    exp_sum = torch.logsumexp(inputs,dim=-1,keepdim=False)
    targets = targets.unsqueeze(1)
    result = - inputs.gather(dim=-1,index=targets).squeeze(-1) + exp_sum
    return torch.mean(result)
    

if __name__ == "__main__":
    x = torch.randn(3,3)
    y = torch.tensor([1,2,2])
    
    inputs = torch.tensor(
            [
                [
                    [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                    [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                    [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                    [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
                ],
                [
                    [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                    [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                    [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                    [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
                ],
            ]
        )
    targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
    inputs = inputs.view(-1, inputs.size(-1))
    targets = targets.view(-1)

    cross_entropy(inputs,targets)

