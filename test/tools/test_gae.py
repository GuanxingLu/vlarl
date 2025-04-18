import torch

local_rollout_batch_size = 1
num_steps = 8
# gamma = 0.99
# lam = 0.95
gamma = 1.0
lam = 1.0

device = torch.device("cuda")
scores = torch.zeros(num_steps, local_rollout_batch_size, device=device, dtype=torch.float32)
scores[4, :] = 1.0
values = torch.zeros(num_steps, local_rollout_batch_size, device=device, dtype=torch.float32)
dones = torch.zeros(num_steps, local_rollout_batch_size, device=device, dtype=torch.bool)
dones[5, :] = True
dones_next = torch.zeros(local_rollout_batch_size, device=device, dtype=torch.bool)

with torch.no_grad():    # TODO: optimize this
    next_value = torch.zeros(local_rollout_batch_size, device=device)

    lastgaelam = 0
    advantages = torch.zeros_like(scores).to(device)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - dones_next.float()  # Convert boolean to float
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1].float()
            nextvalues = values[t + 1]
        delta = scores[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    returns = advantages + values
torch.cuda.empty_cache()

print(f"{scores.reshape(-1)=}")
print(f"{advantages.reshape(-1)=}")
print(f"{returns.reshape(-1)=}")