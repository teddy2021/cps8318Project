import Network
import torch

torch.set_default_tensor_type(
    torch.cuda.FloatTensor if torch.cuda.is_available() else
    torch.FloatTensor
)
net = Network.SegNetwork("./seg_data_files.csv", 1)
results = net.forward()
print(f"After one epoch, the network has {results[1]} loss")
