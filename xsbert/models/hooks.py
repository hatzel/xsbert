from typing import Tuple
import torch


# helpers

def interpolate_reference_embedding(embedding: torch.tensor, N: int):
        assert embedding.shape[0] == 2
        s = 1 / N
        device = embedding.device
        x, r = embedding[0], embedding[-1].unsqueeze(0)
        a = torch.arange(1, 0, -s).unsqueeze(1).unsqueeze(1).to(device)
        g = r + a * (x - r)
        g = torch.cat([g, r])
        return g

def repeat_reference_input(inpt: torch.tensor, N: int):
    x, r = inpt[0].unsqueeze(0), inpt[-1].unsqueeze(0)
    d_repeat = (N,) + (1,) * (len(x.shape) - 1)
    x = x.repeat(d_repeat)
    return torch.cat([x, r])


# roberta 

def roberta_interpolation_hook(N: int, outputs: list):
    def hook(model, inpt):
        g = interpolate_reference_embedding(inpt[0], N=N)
        outputs.append(g)
        return (g,) + inpt[1:]
    return hook

 
 # mpnet

def mpnet_interpolation_hook(N: int, outputs: list):
    def hook(model, inpt):
        g = interpolate_reference_embedding(inpt[0], N=N)
        outputs.append(g)
        p = repeat_reference_input(inpt[3], N=N)
        return (g, inpt[1], inpt[2], p)
    return hook

def mpnet_reshaping_hook(N: int):
     def hook(model, inpt):
          p = repeat_reference_input(inpt[3], N=N)
          return inpt[:3] + (p,)
     return hook


def mistral_interpolation_hook(N: int, outputs: list):
    def hook(model, inpt: Tuple[torch.Tensor]):
        # input is a tuple with a single torch Tensor
        # The torch tensor has the shape [batch, length, embedding]
        # Batch size is always 2: the input embedding and the semenatically neutral reference embedding
        g = interpolate_reference_embedding(inpt[0], N=N)
        print("Intermediats length", len(outputs))
        print("Intermediats inner shape", [x.shape for x in outputs])
        outputs.append(g)
        return (g,)
    return hook