import torch

device = torch.device("cuda")

g_padded_response_ids=[[31758, 31886, 31888, 31856, 31852, 31914, 31872, 2]]

g_vllm_responses = torch.tensor(g_padded_response_ids, device=device, dtype=torch.bfloat16)

print(g_vllm_responses)

# tensor([[3.1744e+04, 3.1872e+04, 3.1872e+04, 3.1872e+04, 3.1872e+04, 3.1872e+04,
#          3.1872e+04, 2.0000e+00]], device='cuda:0', dtype=torch.bfloat16)

g_vllm_responses = torch.tensor(g_padded_response_ids, device=device, dtype=torch.float32)

print(g_vllm_responses)

# tensor([[3.1758e+04, 3.1886e+04, 3.1888e+04, 3.1856e+04, 3.1852e+04, 3.1914e+04,
#          3.1872e+04, 2.0000e+00]], device='cuda:0')