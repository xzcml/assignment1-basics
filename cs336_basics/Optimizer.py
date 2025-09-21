from collections.abc import Callable, Iterable 
from typing import Optional 
import torch 
import math

class SGD(torch.optim.Optimizer): 
    def __init__(self, params, lr=1e-3): 
        if lr < 0: 
            raise ValueError(f"Invalid learning rate: {lr}") 
        defaults = {"lr": lr} 
        super().__init__(params, defaults)         
    
    def step(self, closure: Optional[Callable] = None): 
        loss = None if closure is None else closure() 
        for group in self.param_groups:  
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]: 
                if p.grad is None: 
                    continue
                state = self.state[p] # Get state associated with p. 
                t = state.get("t", 0) # Get iteration number from the state, or initial value. 
                grad = p.grad.data # Get the gradient of loss with respect to p.  
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place. 
                state["t"] = t + 1 # Increment iteration number.
        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, betas = (0.9,0.95), lr = 0.8, weight_decay = 0.1, eps = 1e-8):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(betas=betas,lr=lr,weight_decay=weight_decay,eps=eps)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1,beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                t = state.get("t",1)
                prev_m = state.get("m",torch.zeros_like(grad))
                prev_v = state.get("v",torch.zeros_like(grad))
                m = beta1 * prev_m + (1-beta1) * grad
                v = beta2 * prev_v + (1-beta2) * grad**2
                lr_t = lr * (math.sqrt(1 - (beta2**t)) / (1 - (beta1**t)))
                p.data -= lr_t * m/(torch.sqrt(v + eps))
                p.data -= weight_decay*lr*p.data
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss




weights = torch.nn.Parameter(5 * torch.randn((10, 10))) 
opt = SGD([weights], lr=1e3)
for t in range(10):  
    opt.zero_grad() # Reset the gradients for all learnable parameters. 
    loss = (weights**2).mean() # Compute a scalar loss value. 
    print(loss.cpu().item())  
    loss.backward() # Run backward pass, which computes gradients. 
    opt.step() # Run optimizer step.

        