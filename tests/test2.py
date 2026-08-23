import torch
from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.tensor([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
], dtype=torch.float)

edge_index = torch.tensor([
    [0, 1, 2],
    [1, 2, 3]
], dtype=torch.long)

data = Data(x=x, edge_index=edge_index)
data = data.to(device)

print(data)
print("Device:", data.x.device)
