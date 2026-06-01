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

import os

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.utils import clip_grad_norm_

from rlinf.algorithms.embodiment import compute_qam_actor_objective
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.q_head import MultiQHead
from rlinf.scheduler import Worker
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


def qam_reduce_ensemble(
    q_values: torch.Tensor,
    reduction: str = "mean_minus_std",
    rho: float = 1.0,
    keepdim: bool = True,
) -> torch.Tensor:
    """Reduce QAM critic ensemble values.

    Args:
        q_values: Tensor shaped ``[B, num_q_heads]``.
        reduction: ``mean``, ``min``, or ``mean_minus_std``.
        rho: Pessimism coefficient for ``mean_minus_std``.
        keepdim: Whether to return ``[B, 1]``.

    Returns:
        Reduced critic value.
    """
    if reduction == "mean":
        return q_values.mean(dim=1, keepdim=keepdim)
    if reduction == "min":
        values, _ = q_values.min(dim=1, keepdim=keepdim)
        return values
    if reduction == "mean_minus_std":
        std = q_values.std(dim=1, keepdim=keepdim, unbiased=False)
        return q_values.mean(dim=1, keepdim=keepdim) - rho * std
    raise ValueError(f"Unsupported QAM ensemble reduction {reduction!r}")


def qam_bootstrap_target(
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    next_q_values: torch.Tensor,
    gamma: float,
    action_horizon: int,
    reduction: str = "mean_minus_std",
    rho: float = 1.0,
    bootstrap_type: str = "standard",
) -> torch.Tensor:
    """Compute QAM TD target from rewards, dones, and target critic heads."""
    rewards_for_bootstrap = rewards.sum(dim=-1, keepdim=True)
    discount = gamma**action_horizon
    q_next = qam_reduce_ensemble(
        next_q_values,
        reduction=reduction,
        rho=rho,
        keepdim=True,
    )

    if bootstrap_type == "always":
        not_done = torch.ones_like(rewards_for_bootstrap, dtype=torch.bool)
    elif bootstrap_type == "standard":
        not_done = ~(terminations.to(torch.bool).any(dim=-1, keepdim=True))
    else:
        raise NotImplementedError(f"{bootstrap_type=} is not supported!")

    return rewards_for_bootstrap + not_done.to(q_next.dtype) * discount * q_next


class EmbodiedQAMFSDPPolicy(EmbodiedSACFSDPPolicy):
    """FSDP worker for plain QAM on embodied policies.

    This worker reuses SAC's replay-buffer setup, swaps SAC's entropy actor
    objective for QAM's trajectory sampling + adjoint-matching objective, and
    owns a frozen peer model ``f_beta_model`` for the behavior velocity field.
    QAM critic heads are plain worker-owned modules instead of FSDP-wrapped
    target-model components.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.f_beta_model = None
        self.q_head_qam = None
        self.target_q_head_qam = None

    def init_worker(self):
        self.setup_model_and_optimizer(initialize_target=False)
        self.setup_sac_components()
        self.soft_update_target_model(tau=1.0)
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
            self._offload_qam_critic_components()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(self.model, mode="default")
            self.f_beta_model = torch.compile(self.f_beta_model, mode="default")

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        """Setup live actor, independent critic heads, and frozen f_beta model."""
        module = self.model_provider_func()
        f_beta_module = self.model_provider_func()

        if self.cfg.actor.model.get("gradient_checkpointing", False):
            self.logger.info("[FSDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
            f_beta_module.gradient_checkpointing_enable()
        else:
            self.logger.info("[FSDP] Gradient checkpointing is disabled")

        if hasattr(module, "freeze_vlm"):
            module.freeze_vlm()
        if hasattr(f_beta_module, "freeze_vlm"):
            f_beta_module.freeze_vlm()
        if hasattr(module, "q_head_qam"):
            module.q_head_qam.requires_grad_(False)
        if hasattr(f_beta_module, "q_head_qam"):
            f_beta_module.q_head_qam.requires_grad_(False)

        self.param_names_need_sync = self._collect_qam_sync_param_names(module)

        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype

        self.f_beta_model = self._strategy.wrap_model(
            model=f_beta_module, device_mesh=self._device_mesh
        )
        self.f_beta_model.requires_grad_(False)
        self.f_beta_model.eval()

        self.use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)
        if self.use_dsrl:
            raise ValueError("QAM worker does not support use_dsrl=True.")

        self.q_head_qam = self._build_qam_q_head().to(
            device=self.device,
            dtype=torch.float32,
        )
        self.target_q_head_qam = self._build_qam_q_head().to(
            device=self.device,
            dtype=torch.float32,
        )
        self.target_q_head_qam.load_state_dict(self.q_head_qam.state_dict())
        self.target_q_head_qam.requires_grad_(False)
        self.target_q_head_qam.eval()

        self.optimizer = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters={},
            filtered_optim_config={},
        )[0]
        critic_optim_cfg = self.cfg.actor.critic_optim
        self.qf_optimizer = torch.optim.Adam(
            self.q_head_qam.parameters(),
            lr=critic_optim_cfg.lr,
            betas=(
                critic_optim_cfg.get("adam_beta1", 0.9),
                critic_optim_cfg.get("adam_beta2", 0.999),
            ),
            eps=critic_optim_cfg.get("adam_eps", 1e-8),
        )

        self.build_lr_schedulers()
        self.grad_scaler = self.build_grad_scaler(
            self.cfg.actor.fsdp_config.grad_scaler
        )

    def setup_sac_components(self):
        """Reuse SAC replay setup, but default QAM target updates to critic-only."""
        super().setup_sac_components()
        self.target_update_type = self.cfg.algorithm.get(
            "target_update_type", "q_head_only"
        )
        assert self.target_update_type in ["all", "q_head_only"], (
            f"{self.target_update_type=} is not suppported!"
        )

    @staticmethod
    def _collect_qam_sync_param_names(module):
        """Collect rollout-sync params while excluding QAM critic-only heads."""
        return [
            name
            for name, param in module.named_parameters(remove_duplicate=False)
            if param.requires_grad and "q_head_qam" not in name
        ]

    def get_rollout_state_dict(self) -> dict:
        """Return trainable actor weights for rollout sync without FSDP export."""
        module = getattr(self.model, "module", self.model)
        module = getattr(module, "_fsdp_wrapped_module", module)
        params = dict(module.named_parameters(remove_duplicate=False))
        buffers = dict(module.named_buffers(remove_duplicate=False))

        state = {}
        for name in self.param_names_need_sync:
            value = params.get(name, buffers.get(name))
            if value is None:
                raise KeyError(f"QAM rollout sync key {name!r} not found")
            state[name] = value.detach()
        return state

    def _build_qam_q_head(self):
        """Build a worker-owned QAM critic head outside FSDP."""
        cfg = self._openpi_config()
        if cfg is not None:
            config_name = getattr(cfg, "config_name")
            action_chunk = int(getattr(cfg, "action_chunk"))
            action_env_dim = int(getattr(cfg, "action_env_dim"))
            hidden_dims = list(getattr(cfg, "qam_q_hidden_dims"))
            num_q_heads = int(getattr(cfg, "qam_num_q_heads"))
        else:
            openpi_cfg = self.cfg.actor.model.get("openpi", {})
            config_name = openpi_cfg.get(
                "config_name",
                self.cfg.actor.model.get("config_name", ""),
            )
            action_chunk = int(
                openpi_cfg.get(
                    "action_chunk",
                    self.cfg.actor.model.get("num_action_chunks", 1),
                )
            )
            action_env_dim = int(
                openpi_cfg.get(
                    "action_env_dim",
                    self.cfg.actor.model.get("action_env_dim", 1),
                )
            )
            hidden_dims = list(openpi_cfg.get("qam_q_hidden_dims", (512, 512)))
            num_q_heads = int(openpi_cfg.get("qam_num_q_heads", 2))

        hidden_size = 2048 if "pi05_" in config_name else 1024
        return MultiQHead(
            hidden_size=hidden_size,
            action_feature_dim=action_chunk * action_env_dim,
            hidden_dims=hidden_dims,
            num_q_heads=num_q_heads,
            output_dim=1,
            train_action_encoder=False,
        )

    def _openpi_config(self):
        """Return the wrapped OpenPI config from the live model when available."""
        model = getattr(self, "model", None)
        if model is not None:
            module = getattr(model, "module", model)
            if hasattr(module, "config"):
                return module.config
            wrapped = getattr(module, "_fsdp_wrapped_module", None)
            if wrapped is not None and hasattr(wrapped, "config"):
                return wrapped.config
        return None

    def qam_flow_action_shape(self, batch_size: int) -> tuple[int, int, int]:
        """Shape used by OpenPI/QAM velocity fields in model action space."""
        openpi_config = self._openpi_config()
        if openpi_config is not None:
            horizon = int(getattr(openpi_config, "action_horizon"))
            action_dim = int(getattr(openpi_config, "action_dim"))
            return batch_size, horizon, action_dim

        model_cfg = self.cfg.actor.model
        openpi_cfg = model_cfg.get("openpi", {})
        horizon = int(
            openpi_cfg.get(
                "action_horizon",
                openpi_cfg.get(
                    "action_chunk",
                    model_cfg.get(
                        "num_action_chunks",
                        model_cfg.get("action_chunk", 1),
                    ),
                ),
            )
        )
        action_dim = int(openpi_cfg.get("action_dim", model_cfg.get("action_dim", 1)))
        return batch_size, horizon, action_dim

    def qam_critic_action_shape(self, batch_size: int) -> tuple[int, int, int]:
        """Shape used by QAM critic/env actions after OpenPI output slicing."""
        openpi_config = self._openpi_config()
        if openpi_config is not None:
            horizon = int(getattr(openpi_config, "action_chunk"))
            action_dim = int(getattr(openpi_config, "action_env_dim"))
            return batch_size, horizon, action_dim

        model_cfg = self.cfg.actor.model
        openpi_cfg = model_cfg.get("openpi", {})
        horizon = int(
            openpi_cfg.get(
                "action_chunk",
                model_cfg.get(
                    "num_action_chunks",
                    model_cfg.get("action_chunk", 1),
                ),
            )
        )
        action_dim = int(
            openpi_cfg.get(
                "action_env_dim",
                model_cfg.get("action_dim", model_cfg.get("action_env_dim", 1)),
            )
        )
        return batch_size, horizon, action_dim

    def qam_action_shape(self, batch_size: int) -> tuple[int, int, int]:
        """Backward-compatible alias for critic/env action shape."""
        return self.qam_critic_action_shape(batch_size)

    def _critic_actions_from_flow(self, actions: torch.Tensor) -> torch.Tensor:
        """Slice OpenPI model-space actions to critic/env action space."""
        _, horizon, action_dim = self.qam_critic_action_shape(actions.shape[0])
        if actions.shape[1] < horizon or actions.shape[-1] < action_dim:
            raise ValueError(
                "QAM flow actions must contain critic action dimensions, got "
                f"{tuple(actions.shape)} but need at least (*, {horizon}, {action_dim})"
            )
        return actions[:, :horizon, :action_dim]

    @staticmethod
    def _infer_batch_size_from_obs(obs: dict) -> int:
        for value in obs.values():
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
        raise ValueError("QAM obs must contain at least one batched tensor.")

    def _sample_qam_ode_actions(self, obs: dict) -> torch.Tensor:
        """Sample target actions with deterministic OpenPI/QAM ODE rollout."""
        action_shape = self.qam_flow_action_shape(self._infer_batch_size_from_obs(obs))
        x_t = torch.randn(action_shape, device=self.device, dtype=self.torch_dtype)
        num_steps = int(self.cfg.algorithm.get("flow_steps", 10))
        h = 1.0 / num_steps
        batch_size = action_shape[0]
        with torch.no_grad():
            for step in range(num_steps):
                timestep = torch.full(
                    (batch_size,),
                    step / num_steps,
                    device=self.device,
                    dtype=x_t.dtype,
                )
                velocity = self.model(
                    forward_type=ForwardType.QAM_VELOCITY,
                    obs=obs,
                    x_t=x_t,
                    timestep=timestep,
                )
                x_t = x_t + h * velocity
        return self._critic_actions_from_flow(x_t).clamp(-1.0, 1.0)

    def _qam_q_values(
        self,
        obs: dict,
        actions: torch.Tensor,
        target: bool = False,
        detach_vlm: bool = True,
    ) -> torch.Tensor:
        """Evaluate QAM online or target critic head from replay obs/actions."""
        q_head = self.target_q_head_qam if target else self.q_head_qam
        q_param = next(q_head.parameters())
        critic_device = q_param.device
        critic_dtype = q_param.dtype

        pooled_z = self.model(
            forward_type=ForwardType.QAM_ENCODE,
            obs=obs,
            detach_vlm=detach_vlm,
        )
        pooled_z = pooled_z.to(device=critic_device, dtype=critic_dtype)
        if actions.dim() == 3:
            actions = actions.reshape(actions.shape[0], -1)
        actions = actions.to(device=critic_device, dtype=critic_dtype)
        return q_head(pooled_z, actions)

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"].to(self.torch_dtype)
        terminations = batch["terminations"]

        with torch.no_grad():
            next_actions = self._sample_qam_ode_actions(next_obs)
            all_qf_next_target = self._qam_q_values(
                obs=next_obs,
                actions=next_actions,
                target=True,
            )
            target_q_values = qam_bootstrap_target(
                rewards=rewards,
                terminations=terminations,
                next_q_values=all_qf_next_target,
                gamma=float(self.cfg.algorithm.gamma),
                action_horizon=next_actions.shape[1],
                reduction=self.cfg.algorithm.get(
                    "qam_target_reduction", "mean_minus_std"
                ),
                rho=float(self.cfg.algorithm.get("qam_pessimism_rho", 1.0)),
                bootstrap_type=self.cfg.algorithm.get("bootstrap_type", "standard"),
            )

        all_data_q_values = self._qam_q_values(
            obs=curr_obs,
            actions=actions,
            target=False,
        )
        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values,
            target_q_values.expand_as(all_data_q_values),
        )
        return critic_loss, {
            "q_data": all_data_q_values.mean().item(),
            "q_target": target_q_values.mean().item(),
        }

    def _make_qam_closures(self, obs: dict):
        """Build trainable actor, frozen behavior, and target critic closures."""

        def f_theta_fn(obs_payload, x_t, timestep):
            return self.model(
                forward_type=ForwardType.QAM_VELOCITY,
                obs=obs_payload,
                x_t=x_t,
                timestep=timestep,
            )

        def f_beta_fn(obs_payload, x_t, timestep):
            return self.f_beta_model(
                forward_type=ForwardType.QAM_VELOCITY,
                obs=obs_payload,
                x_t=x_t,
                timestep=timestep,
            )

        def q_grad_fn(x1):
            critic_action = self._critic_actions_from_flow(x1.detach()).clamp(-1.0, 1.0)
            flat_action = critic_action.reshape(x1.shape[0], -1).detach()
            flat_action.requires_grad_(True)

            q_values = self._qam_q_values(
                obs=obs,
                actions=flat_action,
                target=True,
            )
            q_mean = qam_reduce_ensemble(q_values, reduction="mean", keepdim=True)

            flat_action.grad = None
            q_mean.sum().backward()
            grad = flat_action.grad
            if grad is None:
                raise RuntimeError("Failed to compute QAM terminal action gradient.")

            grad_critic = grad.reshape_as(critic_action)
            grad_flow = torch.zeros_like(x1)
            grad_flow[:, : grad_critic.shape[1], : grad_critic.shape[2]] = (
                grad_critic.to(dtype=grad_flow.dtype)
            )
            return grad_flow.detach()

        return f_theta_fn, f_beta_fn, q_grad_fn

    def soft_update_target_model(self, tau=None):
        """Soft-update only QAM target critic head."""
        if tau is None:
            tau = self.cfg.algorithm.tau

        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_q_head_qam.parameters(),
                self.q_head_qam.parameters(),
            ):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(online_param.data, alpha=tau)

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        obs = batch["curr_obs"]
        action_shape = self.qam_flow_action_shape(self._infer_batch_size_from_obs(obs))
        num_steps = int(self.cfg.algorithm.get("flow_steps", 10))
        inv_temp = self.cfg.algorithm.get("inv_temp", 0.3)
        f_theta_fn, f_beta_fn, q_grad_fn = self._make_qam_closures(obs)

        actor_loss, metrics = compute_qam_actor_objective(
            f_theta_fn=f_theta_fn,
            f_beta_fn=f_beta_fn,
            q_grad_fn=q_grad_fn,
            obs=obs,
            action_shape=action_shape,
            num_steps=num_steps,
            inv_temp=inv_temp,
        )
        qam_metrics = {}
        for key, value in metrics.items():
            metric_key = key.removeprefix("actor/")
            if isinstance(value, torch.Tensor):
                qam_metrics[metric_key] = value.detach().float().cpu().item()
            else:
                qam_metrics[metric_key] = float(value)
        return actor_loss, qam_metrics

    def _offload_qam_critic_components(self):
        """Offload non-FSDP QAM critic heads and optimizer state to CPU."""
        self.q_head_qam.to("cpu")
        self.target_q_head_qam.to("cpu")
        self._strategy.offload_optimizer(self.qf_optimizer)

    def _load_qam_critic_components(self):
        """Load non-FSDP QAM critic heads and optimizer state to this device."""
        self.q_head_qam.to(self.device)
        self.target_q_head_qam.to(self.device)
        self._strategy.onload_optimizer(self.qf_optimizer, self.device)

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self, train_actor: bool = True):
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )

        with self.worker_timer("sample"):
            global_batch = next(self.buffer_dataloader_iter)

        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )

        for i, batch in enumerate(train_micro_batch_list):
            train_micro_batch_list[i] = put_tensor_device(batch, device=self.device)

        self.qf_optimizer.zero_grad()
        gbs_critic_loss = []
        all_critic_metrics = {}
        for batch in train_micro_batch_list:
            critic_loss, critic_metrics = self.forward_critic(batch)
            critic_loss = critic_loss / self.gradient_accumulation
            critic_loss.backward()
            gbs_critic_loss.append(critic_loss.item() * self.gradient_accumulation)
            append_to_dict(all_critic_metrics, critic_metrics)
        all_critic_metrics = {
            f"critic/{key}": np.mean(value) for key, value in all_critic_metrics.items()
        }
        qf_grad_norm = clip_grad_norm_(
            self.q_head_qam.parameters(),
            max_norm=float(self.cfg.actor.critic_optim.clip_grad),
        )
        self.qf_optimizer.step()
        self.qf_lr_scheduler.step()

        metrics_data = {
            "qam/critic_loss": np.mean(gbs_critic_loss),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": qf_grad_norm,
            **all_critic_metrics,
        }

        if self.update_step % self.critic_actor_ratio == 0 and train_actor:
            self.optimizer.zero_grad()
            gbs_actor_loss = []
            all_actor_metrics = {}
            for batch in train_micro_batch_list:
                actor_loss, actor_metrics = self.forward_actor(batch)
                actor_loss = actor_loss / self.gradient_accumulation
                actor_loss.backward()
                gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)
                append_to_dict(all_actor_metrics, actor_metrics)
            all_actor_metrics = {
                f"actor/{key}": np.mean(value)
                for key, value in all_actor_metrics.items()
            }
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            self.lr_scheduler.step()
            metrics_data.update(
                {
                    "qam/actor_loss": np.mean(gbs_actor_loss),
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm,
                    **all_actor_metrics,
                }
            )

        target_update_freq = int(self.cfg.algorithm.get("target_update_freq", 1))
        if (
            self.target_q_head_qam is not None
            and target_update_freq > 0
            and self.update_step % target_update_freq == 0
        ):
            self.soft_update_target_model()

        return metrics_data

    def run_training(self):
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)
            self._load_qam_critic_components()

        min_buffer_size = self.cfg.algorithm.replay_buffer.get(
            "min_buffer_size", 100
        )
        if not self.replay_buffer.is_ready(min_buffer_size):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < "
                f"{min_buffer_size}, skipping training"
            )
            return {}

        train_actor_steps = self.cfg.algorithm.get("train_actor_steps", 0)
        train_actor_steps = max(min_buffer_size, train_actor_steps)
        train_actor = self.replay_buffer.is_ready(train_actor_steps)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        self.f_beta_model.eval()
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            metrics_data = self.update_one_epoch(train_actor=train_actor)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        mean_metric_dict = self.process_train_metrics(metrics)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict

    def save_checkpoint(self, save_base_path, step):
        restore_weight_offload = self.is_weight_offloaded
        restore_optimizer_offload = self.is_optimizer_offloaded
        if restore_weight_offload:
            self.load_param_and_grad(self.device)
        if restore_optimizer_offload:
            self.load_optimizer(self.device)
            self._load_qam_critic_components()

        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=self.optimizer,
            lr_schedulers=self.lr_scheduler,
            save_path=save_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )
        f_beta_save_path = os.path.join(
            save_base_path, "qam_components/f_beta_model"
        )
        os.makedirs(f_beta_save_path, exist_ok=True)
        f_beta_state_dict = self._strategy.get_model_state_dict(
            self.f_beta_model, cpu_offload=False, full_state_dict=True
        )
        torch.save(
            f_beta_state_dict,
            os.path.join(f_beta_save_path, f"checkpoint_rank_{self._rank}.pt"),
        )
        for name, module in (
            ("q_head_qam", self.q_head_qam),
            ("target_q_head_qam", self.target_q_head_qam),
        ):
            save_path = os.path.join(save_base_path, f"qam_components/{name}")
            os.makedirs(save_path, exist_ok=True)
            torch.save(
                module.state_dict(),
                os.path.join(save_path, f"checkpoint_rank_{self._rank}.pt"),
            )
        critic_optim_save_path = os.path.join(
            save_base_path, "qam_components/critic_optimizer"
        )
        os.makedirs(critic_optim_save_path, exist_ok=True)
        torch.save(
            {
                "optimizer": self.qf_optimizer.state_dict(),
                "lr_scheduler": self.qf_lr_scheduler.state_dict(),
            },
            os.path.join(
                critic_optim_save_path,
                f"checkpoint_rank_{self._rank}.pt",
            ),
        )
        buffer_save_path = os.path.join(
            save_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.save_checkpoint(buffer_save_path)

        if restore_weight_offload:
            self.offload_param_and_grad()
        if restore_optimizer_offload:
            self.offload_optimizer()
            self._offload_qam_critic_components()

    def load_checkpoint(self, load_base_path):
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=self.optimizer,
            lr_schedulers=self.lr_scheduler,
            load_path=load_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )
        f_beta_load_path = os.path.join(
            load_base_path, "qam_components/f_beta_model"
        )
        f_beta_state_dict = torch.load(
            os.path.join(f_beta_load_path, f"checkpoint_rank_{self._rank}.pt")
        )
        self._strategy.load_model_with_state_dict(
            self.f_beta_model,
            f_beta_state_dict,
            cpu_offload=False,
            full_state_dict=True,
        )
        for name, module in (
            ("q_head_qam", self.q_head_qam),
            ("target_q_head_qam", self.target_q_head_qam),
        ):
            load_path = os.path.join(
                load_base_path,
                f"qam_components/{name}/checkpoint_rank_{self._rank}.pt",
            )
            if not os.path.exists(load_path):
                raise FileNotFoundError(
                    "QAM checkpoint is missing independent critic head "
                    f"{name!r}: {load_path}"
            )
            state_dict = torch.load(load_path, map_location=self.device)
            module.load_state_dict(state_dict)
        critic_optim_load_path = os.path.join(
            load_base_path,
            f"qam_components/critic_optimizer/checkpoint_rank_{self._rank}.pt",
        )
        if not os.path.exists(critic_optim_load_path):
            raise FileNotFoundError(
                "QAM checkpoint is missing independent critic optimizer: "
                f"{critic_optim_load_path}"
            )
        critic_optim_state = torch.load(
            critic_optim_load_path,
            map_location=self.device,
        )
        self.qf_optimizer.load_state_dict(critic_optim_state["optimizer"])
        self.qf_lr_scheduler.load_state_dict(critic_optim_state["lr_scheduler"])
        buffer_load_path = os.path.join(
            load_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.load_checkpoint(buffer_load_path)
