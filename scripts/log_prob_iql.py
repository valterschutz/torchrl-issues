import torch
from torch import nn
from torchrl.data.tensor_specs import OneHot
from torchrl.modules.distributions.discrete import OneHotCategorical
from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.modules.tensordict_module.common import SafeModule
from torchrl.objectives.iql import DiscreteIQLLoss
from torchrl.objectives import DiscreteSACLoss
from tensordict import TensorDict

n_act, n_obs = 4, 3
batch_size = (2,)
spec = OneHot(
    n_act,
    shape=torch.Size(
        (
            *batch_size,
            n_act,
        )
    ),
)
module = SafeModule(
    nn.Linear(n_obs, n_act), in_keys=["observation"], out_keys=["logits"]
)
actor = ProbabilisticActor(
    module=module,
    in_keys=["logits"],
    out_keys=["action"],
    spec=spec,
    distribution_class=OneHotCategorical,
)
qvalue = SafeModule(
    nn.Linear(n_obs, n_act),
    in_keys=["observation"],
    out_keys=["state_action_value"],
)
value = SafeModule(
    nn.Linear(n_obs, 1),
    in_keys=["observation"],
    out_keys=["state_value"],
)
loss = DiscreteIQLLoss(actor, qvalue, value, action_space="one-hot")
# action = spec.rand(batch_size).long()
action = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.int64)
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
print(data)
loss(data)
