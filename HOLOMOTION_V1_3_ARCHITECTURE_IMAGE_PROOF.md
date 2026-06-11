# HoloMotion v1.3 Architecture Image Proof

This file proves the architecture image against concrete HoloMotion v1.3 source
code and calls out one important flaw in the generated image.

Generated image copied into the workspace:

`artifacts/holomotion_v1_3_motion_tracking_architecture_imagegen.png`

Verdict: the parameter numbers in the image are consistent with the ONNX model,
but the data-flow drawing is ambiguous and partly wrong. In particular, the
image makes it look like the Fine Expert Bank points into the Router. Source
code shows the opposite dependency: the Router selects fine experts, then the
selected expert outputs are weighted and combined with the shared expert output.

## Correct Data Flow

The deployed ONNX actor flow is:

```text
obs [B,522]
  -> obs_normalizer.normalize_only + clamp
  -> obs_embed: 522 -> 2048 -> 512
  -> h [B,1,512]

parallel side path:
obs [B,522]
  -> select actor_ref_* feature indices
  -> router_obs_embed: 429 -> 2048 -> 512
  -> router_h [B,1,512]

inside the single GroupedMoEBlock:
h
  -> norm1
  -> ModernAttention
  -> residual add
  -> norm2
  -> compute_moe_ffn(h_norm2, router_x=router_h)
       shared_experts(h_norm2)
       router(router_h): 512 -> 1024 logits
       top_k=2 expert ids and weights
       selected fine experts process h_norm2, not router_h
       shared_out + sparse_out
  -> residual add

then:
  -> final RMSNorm
  -> action_mu_head: 512 -> 512 -> 29
  -> actions [B,29]
```

So ModernAttention is before the expert FFN. The Reference Router side path
does not replace `h`; it only supplies `router_h` for expert selection.

## Proof From ONNX Wrapper

The ONNX wrapper first normalizes the flat observation, then calls the actor
module with `past_key_values` and `step_idx`.

Source:
`thirdparties/HoloMotion/holomotion/src/modules/agent_modules.py:279`

```python
def forward(
    self,
    obs: torch.Tensor,
    past_key_values: torch.Tensor,
    step_idx: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    actor_obs = obs
    if self.obs_norm_enabled:
        actor_obs = self.obs_normalizer.normalize_only(actor_obs)
        if self.obs_norm_clip > 0.0:
            actor_obs = torch.clamp(
                actor_obs, -self.obs_norm_clip, self.obs_norm_clip
            )
    return self.actor_module(
        actor_obs,
        past_key_values=past_key_values,
        current_pos=step_idx,
    )
```

This proves the image boxes:

- `ONNX Inputs`
- `Normalize + Clamp`
- feed into the actor module

## Proof Of Observation Embedding And Router Side Path

The actor's ONNX inference path embeds the full normalized observation into
`h`, and separately computes `router_h` from selected reference-only features.

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:2035`

```python
# Embedding [B, D] -> [B, 1, D]
h = self.obs_embed(x)[:, None, :]  # [1, 1, 512]
router_h = self._compute_router_hidden(x)
if router_h is not None:
    router_h = router_h[:, None, :]
```

The full observation embedding is built as a two-layer MLP.

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:679`

```python
self.obs_embed = nn.Sequential(
    nn.Linear(obs_in, self.obs_embed_mlp_hidden),
    nn.SiLU(),
    nn.Linear(self.obs_embed_mlp_hidden, self.d_model),
)
```

For the v1.3 config:

```text
obs_in = 522
obs_embed_mlp_hidden = 2048
d_model = 512
```

So the image label `522 -> 2048 -> 512` is correct.

The reference-router side path first selects the reference feature indices and
then runs a separate two-layer MLP.

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:2183`

```python
self.router_obs_embed = nn.Sequential(
    nn.Linear(self.router_input_dim, self.router_embed_mlp_hidden),
    nn.SiLU(),
    nn.Linear(self.router_embed_mlp_hidden, self.d_model),
)
```

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:2194`

```python
def _compute_router_hidden(self, x: torch.Tensor) -> torch.Tensor | None:
    if x.shape[-1] != int(self.obs_input_dim or self.input_dim):
        raise ValueError(
            "Reference-routed policy expected flat obs dim "
            f"{int(self.obs_input_dim or self.input_dim)}, got {x.shape[-1]}."
        )
    router_idx = self._router_feature_indices
    if router_idx.device != x.device:
        router_idx = router_idx.to(x.device)
    router_obs = torch.index_select(x, dim=x.ndim - 1, index=router_idx)
    return self.router_obs_embed(router_obs)
```

For the v1.3 config:

```text
router_input_dim = 429
router_embed_mlp_hidden = 2048
d_model = 512
```

So the image label `429 -> 2048 -> 512` is correct.

## Why 512 Router Dims Can Select 1024 Fine Experts

The reference-router MLP outputs a 512-dimensional feature vector, not an expert
vector. The next layer is a linear classifier/scorer over the 1024 fine experts.

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:3975`

```python
self.router = nn.Linear(d_model, num_fine_experts, bias=False)
```

For v1.3:

```text
d_model = 512
num_fine_experts = 1024
```

Therefore:

```text
router_h [B,1,512] -> Linear(512,1024) -> logits [B,1,1024]
```

The ONNX initializer for this appears as `onnx::MatMul_647` with shape
`[512, 1024]`, because ONNX MatMul stores the matrix in input-by-output order
for this exported operation. It has:

```text
512 * 1024 = 524,288 parameters
```

So there is no mismatch: 512 is the router feature dimension; 1024 is the number
of expert scores produced from that feature.

## Proof That Attention Comes Before The MoE Expert FFN

The single `GroupedMoEBlock` runs attention first, then normalizes and runs the
MoE feed-forward path.

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:2071`

```python
h_norm = layer.norm1(h)

# Attention
attn_out, new_k_cache, new_v_cache = (
    layer.attn.forward_single_token(
        x=h_norm,
        cos=cos,
        sin=sin,
        k_cache=k_cache,
        v_cache=v_cache,
        new_len=new_len,
        insert_pos=insert_pos,
    )
)

h = h + attn_out

# FFN / MoE
h_norm2 = layer.norm2(h)

if isinstance(layer, GroupedMoEBlock):
    if export_routing_debug:
        ffn_out, topk_idx, router_logits = layer.compute_moe_ffn(
            h_norm2,
            router_x=router_h,
            return_routing_debug=True,
        )
        routing_debug_outputs.extend([topk_idx, router_logits])
    else:
        ffn_out = layer.compute_moe_ffn(h_norm2, router_x=router_h)
else:
    # Dense MLP
    ffn_out = layer.mlp_dropout(layer.mlp(h_norm2))
h = h + ffn_out
```

This proves the correct ordering:

```text
norm1 -> ModernAttention -> residual add -> norm2 -> MoE FFN -> residual add
```

The image should show ModernAttention before the Fine Expert Bank and Shared
Expert, not as a peer with unclear ordering.

## Proof Of Router, Fine Expert, And Shared Expert Flow

Inside `compute_moe_ffn`, the shared expert processes the token hidden state
`x` directly. The router uses `router_x` when provided. In this model,
`router_x` is `router_h` from the reference-router side path.

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:4396`

```python
def compute_moe_ffn(
    self,
    x: torch.Tensor,
    router_x: torch.Tensor | None = None,
    *,
    return_routing_debug: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, D = x.shape
    ...
    # 1. Shared Experts (Dense Path)
    shared_out = self.shared_experts(x)

    # 2. Router (Gating)
    router_input = x if router_x is None else router_x
    if router_input.shape != x.shape:
        raise ValueError(
            "router_x shape must match x shape in compute_moe_ffn: "
            f"x={tuple(x.shape)}, router_x={tuple(router_input.shape)}"
        )
    if self.freeze_router:
        with torch.no_grad():
            logits = self.router(router_input)
    else:
        logits = self.router(router_input)
```

The router logits are converted into top-k expert ids and scores.

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:4430`

```python
if self.routing_score_fn == "softmax":
    choice_logits = logits_fp32
    if bias_fp32 is not None:
        choice_logits = choice_logits + bias_fp32
    choice_scores = choice_logits
    _, topk_idx = torch.topk(choice_scores, self.top_k, dim=-1)
    dense_distribution = torch.softmax(logits_fp32, dim=-1)
    if torch.onnx.is_in_onnx_export():
        selected_probs = dense_distribution.gather(-1, topk_idx)
    else:
        selected_logits = logits_fp32.gather(-1, topk_idx)
        log_z = torch.logsumexp(logits_fp32, dim=-1, keepdim=True)
        selected_probs = torch.exp(selected_logits - log_z)
    topk_scores = selected_probs / selected_probs.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-20)
```

Then the selected fine experts process `x`, not `router_x`.

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:4494`

```python
# 3. Sparse Experts Computation (Grouped MM / ONNX Loop)
sparse_out = self._compute_sparse_experts(x, topk_idx, topk_scores)

# 4. Combine
output = shared_out + sparse_out
output = self.mlp_dropout(output)
```

This proves the correct dependency:

```text
router_h -> router -> topk_idx/topk_scores
x -> selected fine experts -> sparse_out
x -> shared expert -> shared_out
shared_out + sparse_out -> MoE output
```

The Fine Expert Bank should not point into the Router. Router points to expert
selection.

## What "Gate/Up" And "Down" Mean

This is an MLP, but it is a SwiGLU MLP. SwiGLU uses two first-stage projections:

- `gate`: produces a 256-dimensional gate vector
- `up`: produces a 256-dimensional value vector
- combine: `SiLU(gate) * up`, still 256 dimensions
- `down`: projects 256 back to the model dimension 512

For the shared expert, the code uses a normal `nn.Linear` implementation:

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:4647`

```python
class DeepseekV3MLP(nn.Module):
    """SwiGLU MLP with fused gate+up projection for efficiency."""

    def __init__(self, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # Fused gate and up projection: outputs [gate, up] concatenated
        self.gate_up_proj = nn.Linear(
            self.hidden_size,
            2 * self.intermediate_size,
            bias=True,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=True,
        )
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., hidden_size]
        gate_up = self.gate_up_proj(x)  # [..., 2 * intermediate_size]
        gate, up = gate_up.chunk(2, dim=-1)  # each [..., intermediate_size]
        return self.down_proj(self.act_fn(gate) * up)
```

For v1.3:

```text
hidden_size = d_model = 512
intermediate_size = int(d_model * ff_mult) = int(512 * 0.5) = 256
gate_up output = 2 * 256 = 512
gate vector = 256
up vector = 256
SiLU(gate) * up = 256
down output = 512
```

So the first stage does not mean "up converts 512 to 512" by itself. It is a
fused matrix that computes two 256-dimensional vectors at once.

## Proof Of Fine Expert Tensor Shapes

The fine experts are not stored as 1024 separate `nn.Linear` modules. They are
stored as grouped matrices for efficient top-k expert execution.

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:3972`

```python
self.norm2 = RMSNorm(d_model)
self.intermediate_dim = int(d_model * ff_mult)

self.router = nn.Linear(d_model, num_fine_experts, bias=False)
self._apply_freeze_router_state()

# Gate + Up (Combined)
self.gate_up_proj = nn.Parameter(
    torch.empty(
        num_fine_experts, self.d_model, 2 * self.intermediate_dim
    )
)
# Down
self.down_proj = nn.Parameter(
    torch.empty(num_fine_experts, self.intermediate_dim, self.d_model)
)
```

For v1.3:

```text
num_fine_experts = 1024
d_model = 512
ff_mult = 0.5
intermediate_dim = 256

gate_up_proj shape = [1024, 512, 2 * 256] = [1024, 512, 512]
down_proj shape = [1024, 256, 512]
```

The matrix convention is documented by `_grouped_linear`.

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:5143`

```python
def _grouped_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    offs: torch.Tensor | None = None,
) -> torch.Tensor:
    """input: [Total_Tokens, In_Dim]
    weight: [Num_Experts, In_Dim, Out_Dim]
    """
    orig_dtype = input.dtype
    if input.dtype != weight.dtype:
        input = input.to(weight.dtype)
    out = torch._grouped_mm(input, weight, offs=offs)
```

The ONNX export path uses selected expert matrices directly:

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:4333`

```python
def _compute_with_topk_selection(
    self,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_scores: torch.Tensor,
) -> torch.Tensor:
    ...
    x_tokens = x.reshape(N, D)
    idx = topk_idx.reshape(N, K)
    scores = topk_scores.reshape(N, K)

    x_rep = x_tokens[:, None, :].expand(N, K, D).reshape(N * K, D)
    idx_flat = idx.reshape(N * K)
    ...
    gate_up_w = self.gate_up_proj.index_select(0, idx_flat)
    gate_up_out = torch.bmm(x_rep.unsqueeze(1), gate_up_w).squeeze(1)

    x1, x2 = gate_up_out.chunk(2, dim=-1)
    hidden = F.silu(x1) * x2

    down_w = self.down_proj.index_select(0, idx_flat)
    sparse_flat = torch.bmm(hidden.unsqueeze(1), down_w).squeeze(1)
    ...
    sparse = sparse_flat.view(N, K, D)
    weighted = sparse * scores.to(sparse.dtype).unsqueeze(-1)
    out = weighted.sum(dim=1)
    return out.view(B, T, D)
```

This proves the dimensions:

```text
x_rep:        [N*K, 512]
gate_up_w:   [N*K, 512, 512]
gate_up_out: [N*K, 512]
x1, x2:      [N*K, 256] and [N*K, 256]
hidden:      [N*K, 256]
down_w:      [N*K, 256, 512]
sparse_flat: [N*K, 512]
weighted sum over K=2 selected experts -> [N, 512]
```

The shapes `[1024,512,512]` and `[1024,256,512]` are therefore true and
consistent with the source. The second dimension of `down_proj` is 256 because
the SwiGLU hidden vector after `SiLU(gate) * up` is 256.

## Proof Of ModernAttention Parameters

The attention module uses separate Q, fused KV, output projection, Q/K RMSNorm,
and a headwise gate.

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:4735`

```python
# Fused projections: Q separate, KV fused (for GQA efficiency)
self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
self.kv_proj = nn.Linear(
    d_model, 2 * self.n_kv_heads * self.head_dim, bias=False
)
self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

if self.use_qk_norm:
    self.q_norm = RMSNorm(self.head_dim)
    self.k_norm = RMSNorm(self.head_dim)

if self.use_gated_attn:
    if self.gated_attn_type == "headwise":
        # Qwen3-style: one gate scalar per head [B, T, n_heads]
        self.gate_proj = nn.Linear(d_model, n_heads, bias=False)
```

For v1.3:

```text
d_model = 512
n_heads = 8
n_kv_heads = 4
head_dim = 512 / 8 = 64

q_proj:    512 * (8 * 64) = 262,144
kv_proj:   512 * (2 * 4 * 64) = 262,144
o_proj:    (8 * 64) * 512 = 262,144
gate_proj: 512 * 8 = 4,096
q_norm:    64
k_norm:    64
```

The image's `ModernAttention` parameter subtotal of 790,528 counts the four
projection matrices:

```text
262,144 + 262,144 + 262,144 + 4,096 = 790,528
```

The Q/K norm scales are counted in the separate image note
`MoE norms + Q/K norms: 1,152 params`.

## Proof Of Shared Expert

The shared expert is dense and always runs on `x`. It is a SwiGLU MLP with
`hidden_size=512` and `intermediate_size=256`.

Construction:

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:3989`

```python
self.shared_experts = DeepseekV3MLP(
    hidden_size=d_model,
    intermediate_size=int(d_model * ff_mult * num_shared_experts),
)
```

Execution:

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:4408`

```python
# 1. Shared Experts (Dense Path)
shared_out = self.shared_experts(x)
```

Parameters:

```text
gate_up weight: 512 * 512 = 262,144
gate_up bias: 512
down weight: 256 * 512 = 131,072
down bias: 512
total = 394,240
```

The shared expert does not feed the router. It is parallel to the sparse expert
path and is added to `sparse_out`.

## Proof Of Final Norm And Action Head

After the single MoE block, the ONNX inference path runs final RMSNorm and then
the action mean head.

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:2108`

```python
h = self.norm_f(h)
action = self.action_mu_head(h[:, 0, :])
present_key_values = torch.stack(present_key_values_list, dim=0)
```

Construction:

Source:
`thirdparties/HoloMotion/holomotion/src/modules/network_modules.py:746`

```python
self.norm_f = RMSNorm(self.d_model)
self.action_mu_head = nn.Sequential(
    nn.Linear(self.d_model, self.d_model),
    nn.SiLU(),
    nn.Linear(self.d_model, self.output_dim),
)
```

For v1.3:

```text
norm_f: 512
action_mu_head:
  512 * 512 + 512 = 262,656
  29 * 512 + 29 = 14,877
  total = 277,533
```

## Image Corrections Needed

The generated image should be corrected as follows:

1. Replace any arrow from `Fine Expert Bank` to `Router` with:

```text
Reference Router Side Path -> Router -> top_k expert ids/scores -> Fine Expert Bank
```

2. Make the main residual stream explicit:

```text
Observation Embed MLP -> norm1 -> ModernAttention -> + residual -> norm2
```

3. Make the MoE FFN explicit:

```text
norm2 output x -> Shared Expert -> shared_out
norm2 output x -> selected Fine Experts -> sparse_out
Router output only selects and weights selected Fine Experts
shared_out + sparse_out -> MoE FFN output
```

4. Put `Final RMSNorm` and `Action Mean Head` after the MoE residual output.

This is the unambiguous source-backed architecture. The current image is useful
as a parameter summary, but it should not be treated as a precise data-flow
diagram until those arrows are fixed.
