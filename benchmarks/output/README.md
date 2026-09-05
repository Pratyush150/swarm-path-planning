# Benchmark output

Everything in this directory is generated. Nothing is hand-drawn or hand-edited.

```
python3 benchmarks/run.py            # results.csv, tables.md, success/runtime charts
python3 benchmarks/make_figures.py   # animations, morph renders, space-time diagram
```

Most of the output is ignored by git. The handful of small optimised figures the
README displays is committed:

| file | produced by |
|---|---|
| `swarm-demo.gif`, `swarm-demo.mp4`, `swarm-filmstrip.png` | `make_figures.py --animation` |
| `formation-morph.gif`, `formation-morph.mp4`, `formation-morph.png` | `make_figures.py --morph` |
| `assignment-comparison.png` | `make_figures.py --morph` |
| `conflict-spacetime.png` | `make_figures.py --conflict` |
| `success-rate.png`, `runtime-distribution.png` | `make_figures.py --charts` (needs `results.csv`) |

The MP4s are transcoded from the GIFs when `ffmpeg` is available; they are a
tenth of the size and are what to use in a slide or a video.

`results.csv` and `tables.md` are not committed: they are the measurement, and
they belong to the machine that produced them. The tables in the README are
pasted from a run whose machine and time budget are stated there.
