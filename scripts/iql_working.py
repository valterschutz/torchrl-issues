import torch
from torch import nn
from torch.nn import functional as F
from torchrl.data import Categorical
from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.modules.tensordict_module.common import SafeModule
from torchrl.objectives.iql import IQLLoss
from tensordict import TensorDict

n_act, n_obs = 4, 3
action_spec = Categorical(n=n_act, shape=torch.Size(()))
net = nn.Sequential(nn.Linear(n_obs, n_act))
module = SafeModule(net, in_keys=["observation"], out_keys=["logits"])
actor = ProbabilisticActor(
    module=module,
    in_keys=["logits"],
    spec=action_spec,
    distribution_class=torch.distributions.Categorical,
)


class QValueClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(n_obs + n_act, 1)

    def forward(self, obs, act):
        return self.linear(torch.cat([obs, F.one_hot(act, n_act)], -1))


qvalue = SafeModule(
    QValueClass(),
    in_keys=["observation", "action"],
    out_keys=["state_action_value"],
)

value = SafeModule(
    nn.Linear(n_obs, 1),
    in_keys=["observation"],
    out_keys=["state_value"],
)

loss = IQLLoss(actor, qvalue, value)
batch = [2]
action = action_spec.rand(batch)
data = TensorDict(
    {
        "observation": torch.randn(*batch, n_obs),
        "action": action,
        ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
        ("next", "terminated"): torch.zeros(*batch, 1, dtype=torch.bool),
        ("next", "reward"): torch.randn(*batch, 1),
        ("next", "observation"): torch.randn(*batch, n_obs),
    },
    batch,
)

loss(data)
