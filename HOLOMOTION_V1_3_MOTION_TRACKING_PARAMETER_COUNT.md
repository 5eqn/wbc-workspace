# HoloMotion v1.3 Motion Tracking Parameter Count

This document counts the downloaded HoloMotion v1.3 motion-tracking inference
model:

- Downloaded model: `assets/HoloMotion_models/HoloMotion_motion_tracking_model/exported/motion_tracking_model.onnx`
- Downloaded config: `assets/HoloMotion_models/HoloMotion_motion_tracking_model/config.yaml`
- Deployment copy: `thirdparties/HoloMotion/deployment/unitree_g1_ros2_29dof/src/models/motion_tracking_model/...`
- Source tree inspected: `thirdparties/HoloMotion`, `git describe --tags --always --dirty` = `v1.3.0-4-g239d33e`

`cmp` reports that the downloaded config/model and deployment config/model are
byte-identical. The count below is therefore the exact deployed motion-tracking
ONNX model, not a reconstructed guess.

## Result

The ONNX model has 30 float initializers with a total of **408,692,401 scalar
values**.

Of these, **1,044 scalars** are exported observation-normalizer constants
(`_mean` and the folded division denominator). The remaining **408,691,357
scalars** are actor-network weights, biases, and RMSNorm scales. The exported
ONNX does not contain the PPO critic, optimizer state, or stochastic policy
`log_std`; it is the deterministic deployment actor wrapped by
`PPOTFActorOnnxModule.forward`.

## Architecture Source

The model config sets the actor type and dimensions in
`assets/HoloMotion_models/HoloMotion_motion_tracking_model/config.yaml`:

- `robot.actions_dim: 29` at line 195.
- `modules.actor.type: ReferenceRoutedGroupedMoETransformerPolicy` at line 1579.
- `num_fine_experts: 1024`, `num_shared_experts: 1`, `top_k: 2` at lines 1581-1583.
- `obs_embed_mlp_hidden: 2048`, `router_embed_mlp_hidden: 2048`, `d_model: 512`,
  `n_heads: 8`, `n_kv_heads: 4`, `n_layers: 1`, `ff_mult: 0.5`,
  `max_ctx_len: 32` at lines 1590-1601.
- Actor observation schema at lines 1614-1635.
- `dense_layer_at_first: false` at line 1637, so the single transformer layer is
  a `GroupedMoEBlock`, not a dense `ModernTransformerBlock`.

The construction path is:

- `PPOTF._select_actor_wrapper_cls` maps
  `ReferenceRoutedGroupedMoETransformerPolicy` to `PPOTFRefRouterActor`
  (`thirdparties/HoloMotion/holomotion/src/algo/ppo_tf.py:45`).
- `PPOTF` infers the flat actor observation dimension and sets
  `input_dim_override` (`ppo_tf.py:984` and `ppo_tf.py:1021`).
- `PPOTFRefRouterActor` infers router features from all flattened observation
  terms whose leaf name starts with `actor_ref_`, then writes
  `router_input_dim` and `router_feature_indices`
  (`agent_modules.py:1554`, `agent_modules.py:1577`, `agent_modules.py:1618`).
- `PPOActor` creates `EmpiricalNormalization` when `obs_norm.enabled` is true
  and then instantiates the configured actor network (`agent_modules.py:483`,
  `agent_modules.py:493`).
- `PPOTFActorOnnxModule.forward` applies `obs_normalizer.normalize_only`, clamps,
  and calls `actor_module` (`agent_modules.py:279`).

## Input Dimensions

The ONNX input is `obs: [1, 522]`.

Current actor terms:

| Term | Dim |
|---|---:|
| `actor_ref_gravity_projection_cur` | 3 |
| `actor_ref_base_linvel_cur` | 3 |
| `actor_ref_base_angvel_cur` | 3 |
| `actor_ref_dof_pos_cur` | 29 |
| `actor_ref_root_height_cur` | 1 |
| `actor_projected_gravity` | 3 |
| `actor_rel_robot_root_ang_vel` | 3 |
| `actor_dof_pos` | 29 |
| `actor_dof_vel` | 29 |
| `actor_last_action` | 29 |
| Current subtotal | 132 |

Future actor terms use `obs.n_fut_frames: 10`:

| Term | Per-frame dim | Total dim |
|---|---:|---:|
| `actor_ref_dof_pos_fut` | 29 | 290 |
| `actor_ref_root_height_fut` | 1 | 10 |
| `actor_ref_gravity_projection_fut` | 3 | 30 |
| `actor_ref_base_linvel_fut` | 3 | 30 |
| `actor_ref_base_angvel_fut` | 3 | 30 |
| Future subtotal | 39 | 390 |

Flat actor observation dimension: `132 + 390 = 522`.

The router sees only `actor_ref_*` features:

- Current reference features: `3 + 3 + 3 + 29 + 1 = 39`
- Future reference features: `390`
- Router input dimension: `39 + 390 = 429`

## Parameter Table

| ONNX initializer | Shape | Scalars | Source code and formula |
|---|---:|---:|---|
| `obs_normalizer._mean` | `[1, 522]` | 522 | `EmpiricalNormalization.register_buffer("_mean", ...)` at `network_modules.py:78`; shape is flat obs dim. |
| `onnx::Div_565` | `[1, 522]` | 522 | Exported denominator from `normalize_only: (x - _mean) / (_std + eps)` at `network_modules.py:111`; shape is flat obs dim. |
| `actor_module.obs_embed.0.weight` | `[2048, 522]` | 1,069,056 | `GroupedMoETransformerPolicy.obs_embed` first `nn.Linear(obs_in, obs_embed_mlp_hidden)` at `network_modules.py:679`; `2048 * 522`. |
| `actor_module.obs_embed.0.bias` | `[2048]` | 2,048 | Bias of the same first obs embedding linear. |
| `actor_module.obs_embed.2.weight` | `[512, 2048]` | 1,048,576 | Second obs embedding `nn.Linear(obs_embed_mlp_hidden, d_model)` at `network_modules.py:682`; `512 * 2048`. |
| `actor_module.obs_embed.2.bias` | `[512]` | 512 | Bias of the second obs embedding linear. |
| `actor_module.layers.0.gate_up_proj` | `[1024, 512, 512]` | 268,435,456 | `GroupedMoEBlock.gate_up_proj` at `network_modules.py:3979`; shape is `num_fine_experts * d_model * (2 * intermediate_dim)`, with `intermediate_dim = int(512 * 0.5) = 256`. |
| `actor_module.layers.0.down_proj` | `[1024, 256, 512]` | 134,217,728 | `GroupedMoEBlock.down_proj` at `network_modules.py:3985`; `1024 * 256 * 512`. |
| `actor_module.layers.0.norm1.weight` | `[512]` | 512 | `self.norm1 = RMSNorm(d_model)` at `network_modules.py:3947`; RMSNorm has one scale vector at `network_modules.py:4677`. |
| `actor_module.layers.0.attn.q_norm.weight` | `[64]` | 64 | `ModernAttention.q_norm = RMSNorm(head_dim)` at `network_modules.py:4742`; `head_dim = d_model / n_heads = 64`. |
| `actor_module.layers.0.attn.k_norm.weight` | `[64]` | 64 | `ModernAttention.k_norm = RMSNorm(head_dim)` at `network_modules.py:4744`. |
| `actor_module.layers.0.norm2.weight` | `[512]` | 512 | `self.norm2 = RMSNorm(d_model)` at `network_modules.py:3972`. |
| `actor_module.layers.0.shared_experts.gate_up_proj.bias` | `[512]` | 512 | Shared expert `DeepseekV3MLP.gate_up_proj` bias at `network_modules.py:4655`; intermediate size is `512 * 0.5 * 1 = 256`, so output bias dim is `2 * 256 = 512`. |
| `actor_module.layers.0.shared_experts.down_proj.bias` | `[512]` | 512 | Shared expert `DeepseekV3MLP.down_proj` bias at `network_modules.py:4660`. |
| `actor_module.norm_f.weight` | `[512]` | 512 | Final `RMSNorm(d_model)` at `network_modules.py:746`. |
| `actor_module.action_mu_head.0.weight` | `[512, 512]` | 262,144 | First action head linear at `network_modules.py:747`; `512 * 512`. |
| `actor_module.action_mu_head.0.bias` | `[512]` | 512 | Bias of first action head linear. |
| `actor_module.action_mu_head.2.weight` | `[29, 512]` | 14,848 | Final action head linear at `network_modules.py:750`; output dim resolves from `robot_action_dim` to 29 in `agent_modules.py:569`. |
| `actor_module.action_mu_head.2.bias` | `[29]` | 29 | Bias of final action head linear. |
| `actor_module.router_obs_embed.0.weight` | `[2048, 429]` | 878,592 | Reference router embedding first linear at `network_modules.py:2183`; `router_input_dim = 429`, hidden dim 2048. |
| `actor_module.router_obs_embed.0.bias` | `[2048]` | 2,048 | Bias of first router embedding linear. |
| `actor_module.router_obs_embed.2.weight` | `[512, 2048]` | 1,048,576 | Reference router embedding second linear at `network_modules.py:2186`; `512 * 2048`. |
| `actor_module.router_obs_embed.2.bias` | `[512]` | 512 | Bias of second router embedding linear. |
| `onnx::MatMul_570` | `[512, 512]` | 262,144 | Exported `ModernAttention.q_proj`, `nn.Linear(d_model, n_heads * head_dim, bias=False)` at `network_modules.py:4736`; `512 * 512`. |
| `onnx::MatMul_576` | `[512, 512]` | 262,144 | Exported `ModernAttention.kv_proj`, `nn.Linear(d_model, 2 * n_kv_heads * head_dim, bias=False)` at `network_modules.py:4737`; `2 * 4 * 64 = 512`. |
| `onnx::MatMul_639` | `[512, 8]` | 4,096 | Exported headwise attention gate, `nn.Linear(d_model, n_heads, bias=False)` at `network_modules.py:4749`; `512 * 8`. |
| `onnx::MatMul_644` | `[512, 512]` | 262,144 | Exported `ModernAttention.o_proj`, `nn.Linear(n_heads * head_dim, d_model, bias=False)` at `network_modules.py:4740`. |
| `onnx::MatMul_645` | `[512, 512]` | 262,144 | Exported shared expert `DeepseekV3MLP.gate_up_proj` weight at `network_modules.py:4655`; ONNX stores the MatMul form of the PyTorch linear weight. |
| `onnx::MatMul_646` | `[256, 512]` | 131,072 | Exported shared expert `DeepseekV3MLP.down_proj` weight at `network_modules.py:4660`; ONNX stores the MatMul-transposed form. |
| `onnx::MatMul_647` | `[512, 1024]` | 524,288 | Exported MoE router `nn.Linear(d_model, num_fine_experts, bias=False)` at `network_modules.py:3975`; ONNX stores the MatMul-transposed form. |

## Subtotals

| Group | Scalars |
|---|---:|
| Observation-normalizer exported constants | 1,044 |
| Observation embedding MLP | 2,120,192 |
| Reference-router embedding MLP | 1,929,728 |
| Fine-expert MoE gate/up/down tensors | 402,653,184 |
| MoE block RMSNorm and Q/K RMSNorm scales | 1,152 |
| MoE attention projections and attention gate | 790,528 |
| MoE router projection | 524,288 |
| Shared expert MLP | 394,240 |
| Final RMSNorm | 512 |
| Action mean head | 277,533 |
| **Total ONNX initializers** | **408,692,401** |

## Verification Command

This is the exact command used to verify the model file:

```bash
python - <<'PY'
from pathlib import Path
import onnx

p = Path("assets/HoloMotion_models/HoloMotion_motion_tracking_model/exported/motion_tracking_model.onnx")
m = onnx.load(str(p), load_external_data=False)
total = 0
for t in m.graph.initializer:
    n = 1
    for d in t.dims:
        n *= int(d)
    total += n
    print(f"{t.name}\t{list(t.dims)}\t{n}")
print("total", total)
PY
```

It reports `initializers = 30` and `total = 408692401`. The ONNX graph inputs
are `obs: [1, 522]`, `past_key_values: [1, 2, 1, 32, 4, 64]`, and
`step_idx: [1]`; the action output is `actions: [1, 29]`.
