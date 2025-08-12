import torch
from torch import nn
from torchrl.data.tensor_specs import OneHot
from torchrl.modules.distributions import NormalParamExtractor, OneHotCategorical
from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
from torchrl.modules.tensordict_module.common import SafeModule
from torchrl.objectives.sac import DiscreteSACLoss
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

n_act, n_obs = 4, 3
batch_size = (2,)
spec = OneHot(n_act, shape=torch.Size((*batch_size, n_act)))
module = TensorDictModule(
    nn.Linear(n_obs, n_act), in_keys=["observation"], out_keys=["logits"]
)
actor = ProbabilisticActor(
    module=module,
    in_keys=["logits"],
    out_keys=["action"],
    spec=spec,
    distribution_class=OneHotCategorical,
)
qvalue = TensorDictModule(
    nn.Linear(n_obs, n_act),
    in_keys=["observation"],
    out_keys=["action_value"],
)
loss = DiscreteSACLoss(actor, qvalue, action_space=spec, num_actions=spec.space.n)
# action = spec.rand(batch)
action = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
print(f"{action.shape=}")
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
loss(data)
