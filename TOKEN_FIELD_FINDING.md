# The token field was never active

`hac26/field.py::TokenField` initialised both token parameters to zeros:

```python
self.p = nn.Parameter(torch.zeros(n_tokens, 3))
self.z = nn.Parameter(torch.zeros(n_tokens, dim))
```

With every token identical, the attention logits are identical, so `softmax` is
uniform for every `y`, `a @ v(z)` does not depend on `y`, and the `out - out.mean()`
on the last line of `forward` removes what remains. `Delta(y)` is **exactly zero**.

It also stays zero. Identical tokens receive identical gradients, so the
configuration is a symmetric critical point that gradient descent cannot leave. The
tokens never differentiate, no matter how long anything trains.

So `ImplicitBody`, documented as `f(y) = core(y) + s * Delta(y)`, has been running as
`f(y) = core(y)` — a convex core and nothing else.

## Evidence

Measured on a 40-body corpus, 3000 Adam steps, `scripts/fit_shapes.py`'s own protocol:

```
                              at init      after 3038 steps
max |Delta| over 4000 points  4.7e-10      0.0 - 1.5e-08
token position spread                      0.00e+00
token latent spread                        0.00e+00
```

The consequence for training is worse than the representation loss. `fit_shapes.py`
saves `codes = cat(tokens.p, tokens.z)`, 608 numbers per body, and `train_lpd.py`
uses exactly that as `x1`, the target the flow is trained to produce:

```
per-component variance of the codes across 40 bodies:  5.8e-13
max |code_i - code_j| over all pairs:                  9.1e-06
```

Every body has the same code. The flow's regression target is a constant vector
independent of which asteroid it is looking at. It cannot have learned anything
about shape from it, and a network that has learned to emit that constant will
decode to zero correction on every input — which is what the reconstructions show:
`D_rms` 0.006 / 0.007 / 0.007 on models 1-3, essentially independent of the body,
against 0.202 needed for model 3.

## Fix

Initialise the tokens distinct. Positions are spread through the body rather than
jittered around the origin, because a token reaches only `sigma = 0.25 R` and tokens
clustered at the centre cannot describe a waist:

```python
u = torch.randn(n_tokens, 3)
u = u / u.norm(dim=1, keepdim=True)
r = 0.85 * radius * torch.rand(n_tokens, 1) ** (1.0 / 3.0)
self.p = nn.Parameter(u * r)
self.z = nn.Parameter(0.5 * torch.randn(n_tokens, dim))
```

Applied to `hac26/field.py`. Nothing else changed.

## Effect

Same targets, same steps, same optimiser. Deep tail of the corpus (`D_rms >= 0.22`,
11 bodies) plus the public ground truth for model 3, 1200 steps:

```
init        corpus mse   corpus acc    m3 mse    m3 acc   max|Delta|   code var
shipped        0.00976       0.8616   0.00395    0.8702    9.3e-09     1.15e-12
broken         0.00035       0.9477   0.00024    0.9480    3.1e+00     3.05e-01
```

28x lower SDF error on the deep tail, 16x on model 3, and the codes finally carry
per-body information.

Across the full 40-body corpus at 3000 steps, the effect is to flatten the
dependence on concavity that made the representation look capacity-limited:

```
D_rms bin        dead field           patched
                mse    sign acc     mse    sign acc
[0.00,0.02)   0.00137    0.925    0.00018    0.965
[0.02,0.12)   0.00140    0.916    0.00020    0.958
[0.12,0.22)   0.00315    0.900    0.00018    0.964
[0.22,0.50)   0.00976    0.862    0.00038    0.944

corr(D_rms, mse)  +0.716              +0.339
```

The earlier reading — that the field lacks capacity for model-3-grade concavity —
was wrong. It had no capacity at all, for anything non-convex. `figures/token_field_fix.png`
shows the difference: fitted to model 3's ground truth, the shipped field returns a
smooth blob at `D_rms` 0.002, the patched one returns the two lobes and the neck at
`D_rms` 0.203 against the truth's 0.202.

## What this does not establish

The flow solver has not been retrained — that needs `runs/surrogate.pt`, and the
surrogate trains against the mesh forward chain, which needs a GPU. So the claim here
is about the representation and the training target, not about the end-to-end score.
What it predicts is that a retrain now has a target that varies with the body, where
before it did not.

Two other things worth checking before that retrain:

* `TOKEN_SIGMA_FRAC = 0.25` is a fixed reach and was never exercised, since the field
  was inert. It is now load-bearing and nobody has tuned it.
* The convex-stage gap is untouched by any of this. Model 3's convex solve loses
  0.175 against a hull ceiling of 0.8669, and model 2 loses 0.087 against a truth
  that is an exactly convex cube. Neither is a token-field problem.

Reproduce with:

```
python tools/fit_corpus.py prep --corpus new --bodies 40
python tools/fit_corpus.py fit  --corpus new --steps 3000    # repeat until done
python tools/fit_corpus.py report --corpus new
python tools/token_init_test.py            # control
python tools/token_init_test.py --broken   # patched init
python tools/token_init_test.py --report
```

`tests/test_field.py`: 10 of 11 pass with the patch. The eleventh extracts a cube at
128^3 and exhausts memory in a 4 GB sandbox; it was not run before or after.
