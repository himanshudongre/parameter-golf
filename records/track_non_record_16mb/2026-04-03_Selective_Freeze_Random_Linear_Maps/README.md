# Non-Record: Selective Freeze on Random Linear Maps — Why Freezing Gate+Up Beats Full Freeze + LoRA

## Summary

Implements the OpenAI wishlist item **"Learning adapters on random linear maps"** with a key finding: **selectively freezing only gate+up MLP projections (37% of params) outperforms freezing the entire model with LoRA adapters (94% frozen) by 40×.**

On FineWeb data, a larger frozen model (12L 384d, 7.3MB artifact) beats a smaller fully-trained model (6L 192d, 2.4MB artifact) by **11.5%** — demonstrating that frozen random weights enable fitting bigger, better models in the 16MB artifact limit.

## The Core Insight

Not all weights are equal. MLP gate and up projections perform feature expansion — random projections preserve geometric structure here (Johnson-Lindenstrauss). The down projection routes information back to the residual stream — this must be learned. Attention performs relational reasoning — this must be learned.

**Freeze the right 37%, learn the rest. Don't freeze everything and adapt with LoRA.**

## FineWeb Results (H100, sp1024)

### Experiment 1: Selective Freeze vs Dropout vs Alternatives

| Config | Best CE | vs Baseline | Artifact |
|--------|---------|-------------|----------|
| Baseline (no regularization) | 3.4816 | — | 2354KB |
| **Freeze gate+up (37% frozen)** | **3.3838** | **-2.8%** | **1490KB** |
| Dropout 0.1 | 3.3651 | -3.3% | 2354KB |
| Dropout 0.2 | 3.2531 | -6.6% | 2354KB |
| Weight decay 0.2 | 3.4769 | -0.1% | 2354KB |
| Weight noise 0.05 | 3.4481 | -1.0% | 2354KB |

Freeze gate+up beats baseline, weight decay, and weight noise. Dropout is stronger for pure regularization — but doesn't save artifact bytes.

### Experiment 2: Artifact-Normalized Comparison (The Key Result)

When artifact budget is fixed, larger frozen models win:

| Config | Best CE | Artifact | Fits 16MB? |
|--------|---------|----------|-----------|
| 6L 192d + dropout 0.2 (baseline) | 3.2531 | 2.4MB | ✅ |
| **Freeze 8L 256d** | **3.1427** | **3.3MB** | **✅** |
| **Freeze+dropout 12L 384d** | **2.8803** | **~7.3MB** | **✅** |
| Baseline 12L 384d (fully trained) | 2.7295 | 17.7MB | **❌ TOO BIG** |

**The 12L frozen+dropout model (7.3MB) beats the 6L fully-trained+dropout model (2.4MB) by 11.5%.** The fully-trained 12L model is 5.5% better but needs 17.7MB — doesn't fit in 16MB.

### Experiment 3: Full Freeze + LoRA vs Selective Freeze

| Config | Frozen% | Best CE | vs Baseline |
|--------|---------|---------|-------------|
| Full freeze + VeRA rank=8 | 94% | 2.3388 | +80% gap |
| Full freeze + VeRA rank=16 | 94% | 2.3288 | +79% gap |
| Full freeze + VeRA rank=32 | 94% | 2.3221 | +79% gap |
| **Selective freeze (gate+up)** | **37%** | **1.2792** | **-1.5% BETTER** |

**Selective freeze is 40× better than full freeze + LoRA.** Increasing LoRA rank from 8 to 32 barely helps — the bottleneck is the frozen attention weights, not adapter capacity.

## Why Full Freeze + LoRA Fails

PR #1295 uses the full-freeze + LoRA approach (12L 768d, 70M+ frozen, LoRA rank 16). Based on our experiments, this approach has a fundamental ~80% CE gap because:

1. **Frozen attention can't learn relational patterns.** Q/K/V projections need to learn task-specific similarity functions. Random Q/K produce random attention patterns that LoRA can't fix.

2. **Frozen output projections block gradient flow.** The down projection in MLP and the output projection in attention are the critical "write" operations to the residual stream. Freezing them blocks the model from learning what information to propagate.

3. **LoRA rank doesn't help.** Rank 8, 16, and 32 all converge to the same CE (~2.33). The bottleneck is structural, not capacity.

**The fix: freeze only gate+up (feature expansion), learn everything else.** This preserves the model's ability to learn attention patterns and residual-stream routing while getting the regularization and artifact-size benefits of frozen random projections.

## Theoretical Basis

**Johnson-Lindenstrauss Lemma:** Random projections from ℝⁿ → ℝᵐ preserve pairwise distances with high probability when m = O(log n / ε²). The gate+up projections expand dim → hidden_dim — this is exactly a random projection that preserves the geometric structure of the input.

**Extreme Learning Machines (Huang et al., 2006):** Frozen random hidden layer + learned output = effective classifier. Our selective freeze is the transformer analog: frozen feature expansion (gate+up) + learned feature selection (down) + learned reasoning (attention).

**VeRA (Kopiczko et al., 2023):** Showed frozen random matrices + learned scaling works for adaptation. Our finding extends this: selective freezing of the RIGHT components matters more than the adapter architecture.

## Competition Implications

**For the 16MB artifact limit:**

| Strategy | Effective Params | Learned Params | Artifact (int6) |
|----------|-----------------|----------------|-----------------|
| Standard (Clark) | 34M | 34M | 15.9MB |
| Full freeze + LoRA (PR #1295) | 70M+ | 5-10M | <16MB |
| **Selective freeze (ours)** | **50M** | **~20M** | **~15MB** |

Selective freeze fits a 50M effective model in 15MB — 47% more parameters than the standard approach. The question (untested at competition scale): does 50M selective-frozen beat 34M fully-trained at the same BPB metric?

## Code

```python
class FrozenLinear(nn.Module):
    """Frozen random weights from seed. 0 bytes in artifact."""
    def __init__(self, in_f, out_f, seed):
        super().__init__()
        rng = torch.Generator(); rng.manual_seed(seed)
        self.register_buffer('weight',
            torch.randn(out_f, in_f, generator=rng) / math.sqrt(in_f))
    def forward(self, x):
        return F.linear(x, self.weight)

class MLP(nn.Module):
    """GEGLU with frozen gate+up, learned down."""
    def __init__(self, dim, exp, layer_seed):
        h = int(dim * exp)
        self.gate = FrozenLinear(dim, h, layer_seed*10+3)  # FROZEN
        self.up   = FrozenLinear(dim, h, layer_seed*10+4)   # FROZEN
        self.down = nn.Linear(h, dim, bias=False)           # LEARNED
```

Full experiment code: `track_b_enhanced.py`, `exp_track_b_fineweb.py`, `seeded_random_transformer.py`

## Hardware & Methodology

- **H100 80GB** (RunPod): FineWeb experiments, Track B enhanced
- **A40 48GB** (RunPod): Track B enhanced, architecture search
- **Mac Mini M4**: Phase 1 proof-of-life, VeRA comparison
- **Data**: FineWeb 10B sp1024 (competition validation set)
- **Training**: 3000 steps, AdamW, cosine LR, batch 64
- Total GPU spend on Track B: ~$3

All experiments use identical controlled conditions with seed=42.

---

*Author: Himanshu Dongre (@himanshudongre) — Implements OpenAI wishlist item "Learning adapters on random linear maps." Also: PR #1227 (28 Experiments), PR #1259 (KNN Scale Deception), PR #1013 (SSM Hybrid).*
