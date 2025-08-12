import torch
from torch import nn
from torchrl.data import OneHot
from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.modules.tensordict_module.common import SafeModule
from torchrl.objectives import DiscreteIQLLoss
from tensordict import TensorDict

n_act, n_obs = 4, 3
batch_size = torch.Size((2,))
action_spec = OneHot(n=n_act, shape=batch_size + torch.Size((n_act,)))

actor_module = nn.Linear(n_obs, n_act)
actor_tdmodule = SafeModule(actor_module, in_keys=["observation"], out_keys=["logits"])
actor_tdmodule_prob = ProbabilisticActor(
    module=actor_tdmodule,
    in_keys=["logits"],
    spec=action_spec,
    distribution_class=torch.distributions.OneHotCategorical,
)


class QValueClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(n_obs + n_act, 1)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], -1))


qvalue_module = QValueClass()
qvalue_tdmodule = SafeModule(
    QValueClass(),
    in_keys=["observation", "action"],
    out_keys=["state_action_value"],
)


value_module = nn.Linear(n_obs, 1)
value_tdmodule = SafeModule(
    value_module,
    in_keys=["observation"],
    out_keys=["state_value"],
)

loss_tdmodule = DiscreteIQLLoss(actor_tdmodule_prob, qvalue_tdmodule, value_tdmodule)

action = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int64)
data = TensorDict(
    {
        "observation": torch.randn(*batch_size, n_obs),
        "action": action,
        ("next", "done"): torch.zeros(*batch_size, 1, dtype=torch.bool),
        ("next", "terminated"): torch.zeros(*batch_size, 1, dtype=torch.bool),
        ("next", "reward"): torch.randn(*batch_size, 1),
        ("next", "observation"): torch.randn(*batch_size, n_obs),
    },
    batch_size,
)

loss_td = loss_tdmodule(data)
loss = loss_td["loss_actor"] + loss_td["loss_value"] + loss_td["loss_qvalue"]
loss.backward()

print(actor_module.weight.grad is not None)  # True
print(value_module.weight.grad is not None)  # True
print(qvalue_module.net.weight.grad is not None)  # False?
