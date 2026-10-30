"""Own recurrent PPO implementation: dual GAE, contiguous TBPTT, burn-in, clipped losses."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from .activation import require_ready
from .config import Config
from .network import tensor_inputs


@dataclass
class Rollout:
    inputs: dict
    hidden: np.ndarray
    starts: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    ended: np.ndarray
    bootstrap: np.ndarray


def advantages(rollout: Rollout, config: Config) -> tuple[np.ndarray, np.ndarray]:
    require_ready()
    gamma = np.array([config.gamma, config.intrinsic_gamma], dtype=np.float32)
    result = np.zeros_like(rollout.rewards)
    carry = np.zeros_like(rollout.bootstrap)
    next_value = rollout.bootstrap
    for t in reversed(range(len(rollout.actions))):
        alive = (~rollout.ended[t])[:, None]
        delta = rollout.rewards[t] + gamma * next_value * alive - rollout.values[t]
        carry = delta + gamma * config.gae_lambda * alive * carry
        result[t] = carry
        next_value = rollout.values[t]
    return result, result + rollout.values


def optimize(
    model,
    optimizer,
    rollout: Rollout,
    config: Config,
    device: torch.device,
    rng: np.random.Generator,
    progress: float,
) -> dict:
    require_ready()
    model.train()
    adv, returns = advantages(rollout, config)
    fraction = min(1.0, progress / config.intrinsic_anneal_fraction)
    beta = config.intrinsic_start + fraction * (config.intrinsic_end - config.intrinsic_start)
    combined = adv[:, :, 0] + beta * adv[:, :, 1]
    combined = (combined - combined.mean()) / max(float(combined.std()), 1e-8)
    entropy_coefficient = config.entropy_start + progress * (config.entropy_end - config.entropy_start)
    lr = config.learning_rate * (1.0 - progress * (1.0 - config.final_lr_fraction))
    for group in optimizer.param_groups:
        group["lr"] = lr
    length, num_envs = rollout.actions.shape
    sequences = [
        (start, env) for start in range(0, length, config.sequence_length) for env in range(num_envs)
    ]
    logs = []
    applied_batches = 0
    stopped = False
    for _ in range(config.epochs):
        for offset in range(0, len(sequences), config.sequences_per_batch):
            require_ready()
            if offset == 0:
                rng.shuffle(sequences)
            chosen = sequences[offset : offset + config.sequences_per_batch]
            starts = np.array([x[0] for x in chosen])
            envs = np.array([x[1] for x in chosen])
            begin = np.maximum(0, starts - config.burn_in)
            hidden = torch.as_tensor(rollout.hidden[begin, envs], device=device)
            if config.burn_in:
                prefix = starts[None, :] - config.burn_in + np.arange(config.burn_in)[:, None]
                valid = prefix >= 0
                prefix = np.maximum(prefix, 0)
                burn_inputs = tensor_inputs(
                    {k: v[prefix, envs[None, :]] for k, v in rollout.inputs.items()}, device
                )
                with torch.no_grad():
                    _, _, hidden = model.sequence(
                        burn_inputs,
                        hidden,
                        torch.as_tensor(rollout.starts[prefix, envs[None, :]], device=device),
                        torch.as_tensor(valid, device=device),
                    )
            indices = starts[None, :] + np.arange(config.sequence_length)[:, None]
            inputs = tensor_inputs({k: v[indices, envs[None, :]] for k, v in rollout.inputs.items()}, device)
            logits, predicted, _ = model.sequence(
                inputs,
                hidden.detach(),
                torch.as_tensor(rollout.starts[indices, envs[None, :]], device=device),
            )
            distribution = Categorical(logits=logits)
            actions = torch.as_tensor(rollout.actions[indices, envs[None, :]], device=device)
            old_log = torch.as_tensor(rollout.log_probs[indices, envs[None, :]], device=device)
            log_ratio = distribution.log_prob(actions) - old_log
            ratio = log_ratio.exp()
            advantage = torch.as_tensor(combined[indices, envs[None, :]], device=device)
            policy_loss = torch.maximum(
                -advantage * ratio, -advantage * ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
            ).mean()
            old_value = torch.as_tensor(rollout.values[indices, envs[None, :]], device=device)
            target = torch.as_tensor(returns[indices, envs[None, :]], device=device)
            clipped_value = old_value + (predicted - old_value).clamp(-config.value_clip, config.value_clip)
            value_loss = (
                0.5 * torch.maximum((predicted - target).square(), (clipped_value - target).square()).mean()
            )
            entropy = distribution.entropy().mean()
            kl = ((ratio - 1.0) - log_ratio).mean()
            loss = policy_loss + config.value_coefficient * value_loss - entropy_coefficient * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite PPO loss; inspect rewards and checkpoint before resuming")
            logs.append(
                {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy.item(),
                    "approx_kl": kl.item(),
                    "clip_fraction": ((ratio - 1.0).abs() > config.clip_ratio).float().mean().item(),
                }
            )
            if kl.item() > config.target_kl:
                stopped = True
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            applied_batches += 1
        if stopped:
            break
    summary = {key: float(np.mean([item[key] for item in logs])) for key in logs[0]} if logs else {}
    summary.update(
        learning_rate=lr,
        intrinsic_weight=beta,
        entropy_coefficient=entropy_coefficient,
        kl_early_stop=stopped,
        gradient_batches=applied_batches,
        minibatches_considered=len(logs),
    )
    return summary
