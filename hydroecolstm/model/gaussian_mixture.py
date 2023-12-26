
import torch
from torch import nn

class GaussianMixture(nn.Module):
    def __init__(self, input_size: int, hidden_size:int, output_size:int,):
        
        super(GaussianMixture, self).__init__()
        
        # Create two linear layer
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.ReLU = nn.ReLU()
        self.linear_out = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor):
        output = self.ReLU(self.linear_in(x))
        output = self.linear_out(output)

        # split output into mu, sigma and weights
        mu, sigma, pi = output.chunk(3, dim=-1) 
        
        return {'mu': mu, 
                'sigma': torch.exp(sigma) + 1.0e-5, 
                'pi': torch.softmax(pi, dim=-1)}
   
'''    
x = torch.rand(10, 6)
y = x.chunk(2, dim=-1) 
y.shape

input_size=4
hidden_size=5
output_size=6
'''

model = GaussianMixture(4,20,3)
test = model(x)
