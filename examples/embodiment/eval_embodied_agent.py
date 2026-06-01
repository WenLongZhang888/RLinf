# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import hydra
import torch.multiprocessing as mp
from omegaconf import open_dict
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


def _resolve_eval_ckpt_path(cfg) -> None:
    """Use an embodied training checkpoint directory for standalone eval."""
    if cfg.runner.get("ckpt_path", None):
        return
    resume_dir = cfg.runner.get("resume_dir", None)
    if not resume_dir:
        return

    candidates = [
        os.path.join(resume_dir, "actor/model_state_dict/full_weights.pt"),
        os.path.join(resume_dir, "model_state_dict/full_weights.pt"),
    ]
    for ckpt_path in candidates:
        if os.path.exists(ckpt_path):
            with open_dict(cfg):
                cfg.runner.ckpt_path = ckpt_path
            return

    raise FileNotFoundError(
        "runner.resume_dir was provided for eval, but no full model weights "
        f"were found. Checked: {candidates}"
    )


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    cfg.runner.only_eval = True
    cfg = validate_cfg(cfg)
    _resolve_eval_ckpt_path(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )
    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    runner = EmbodiedEvalRunner(
        cfg=cfg,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
