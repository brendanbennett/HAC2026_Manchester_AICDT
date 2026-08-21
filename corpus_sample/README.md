# Corpus sample — 20 bodies from `hac26/shapes_nonconvex.py`

Regenerate exactly with `python tools/make_corpus_sample.py` (n=20, seed=7).
Files are named by rank in ascending hull deficit; `corpus_sample.json` carries
`D_rms`, `D_lo` and `R` per body.

Bodies are in the challenge convention already: rotation axis = z, z spanning
[-1, 1] with both planes touched, xy centred. `R` is the bounding-cylinder radius,
sampled from the published table's distribution and then imposed exactly.

| | |
|---|---|
| `body00`–`body07` | the 40% near-convex quota (`D_rms` 0.000–0.006), kept on purpose |
| `body08`–`body19` | the non-convex strata, `D_rms` 0.054–0.332, all with `D_lo ≥ 0.73` |

Public ground truth for scale: model 1 `D_rms` 0.003, model 2 0.000, model 3 **0.202**
(`D_lo` 0.91). The flow solver currently emits 0.006 / 0.007 / 0.007 respectively.

Note the sample needs `rtree` installed. Without it trimesh's ray engine raises,
`sample_corpus` catches the exception and returns an **empty** corpus with only a
warning line.
