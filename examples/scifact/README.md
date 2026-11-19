# SciFact

Real claims, written by experts, checked against PubMed abstracts. This is the
claim-to-evidence task. The Acme handbook is a separate demo.

## Get the data

The abstracts and the claims are not in git.

```bash
make data-scifact
# or: autorag fetch scifact
```

That downloads the official release from
<https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz>
(Allen Institute for AI, [github.com/allenai/scifact](https://github.com/allenai/scifact))
and writes:

- `docs/` — one Markdown file per abstract, one sentence per paragraph
- `.cache/claims_train.jsonl` and `.cache/claims_dev.jsonl`

The hidden test split is not saved. `docs/` and `.cache/` stay out of git.

## License

The SciFact dataset is released under [CC BY-NC 2.0](https://creativecommons.org/licenses/by-nc/2.0/).
Using the data means agreeing to those terms. See
<https://github.com/allenai/scifact/blob/master/LICENSE.md>.

## Citation

```bibtex
@inproceedings{Wadden2020FactOF,
  title={Fact or Fiction: Verifying Scientific Claims},
  author={David Wadden and Shanchuan Lin and Kyle Lo and Lucy Lu Wang and Madeleine van Zuylen and Arman Cohan and Hannaneh Hajishirzi},
  booktitle={EMNLP},
  year={2020},
}
```

## Check a claim

```bash
autorag tune-claims --source examples/scifact
autorag check-claim "Vitamin D supplementation prevents respiratory infection." --source examples/scifact
```

`tune-claims` chooses a retriever and the NLI cuts on the train split only, then
scores the dev split once. The numbers land in `results.json`. Dev is not used
to pick the winner. The test labels are never loaded.

The verdict model is `cross-encoder/nli-deberta-v3-small` (a public NLI
cross-encoder, CPU, no API key). It needs the `local` extra and PyTorch, which
do not fit a 512 MB free host. On that host `AUTORAG_RUNTIME=lite` keeps
`POST /check-claim` working with an extractive verdict: shared words plus a
negation check, no NLI weights.

## Dev scores

From `results.json`, after one run. The winner was chosen on train macro-F1.
Dev was scored once.

| | Baseline (BM25, always supports) | Tuned (hybrid + NLI) |
| --- | ---: | ---: |
| Accuracy | 0.4133 | 0.4667 |
| Macro-F1 | 0.1950 | 0.4676 |
| Abstract recall@5 | 0.7420 | 0.8269 |
| Sentence recall@5 | 0.4117 | 0.5429 |
| Seconds per claim | 0.0748 | 0.1964 |

Train label counts: supports 332, refutes 173, not enough evidence 304.
Dev: supports 124, refutes 64, not enough evidence 112.

The cuts, chosen on train only, are 0.45 to say supports and 0.65 to say refutes.
Hybrid with 5 sentences won train macro-F1 at 0.4788. BM25 plus NLI scored 0.4305.
Taking 8 sentences scored 0.4560. Dev was not used for that choice.

| Dev label | How many | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| supports | 124 | 0.7963 | 0.3468 | 0.4831 |
| refutes | 64 | 0.3381 | 0.7344 | 0.4631 |
| not enough evidence | 112 | 0.4673 | 0.4464 | 0.4566 |
