# Talking through AutoRAG

Numbers below are from `data/acme/tune_results.json` unless the sentence says Harbor Kitchen.

## 30 seconds

I point this at a folder of documents. It answers questions by quoting the page, and it tries several ways of cutting and searching those pages. It keeps a new way only when the score on a set of known questions goes up. On the fake Acme handbook, "PTO after 3 years" went from the five-year line in the 2025 policy, 25 days, to the three-year line in that same policy, 22 days. The score went from 0.8968 to 0.9435.

## Two minutes

Acme is only the demo. The program does not know the company name. You pass a folder.

First it reads Markdown, text, and PDF. A page can say `supersedes: some-older-id` in the header. That older page is then treated as retired. Nothing in the code looks for the string "Acme" or a particular file name.

Next it splits pages into passages. A passage is a chunk. The blunt split is a window of words, which cuts through tables. The better split keeps a heading with its paragraphs and keeps a table in one piece.

Then each passage becomes a list of numbers, an embedding, so "vacation" can land near "PTO". Vector search finds the nearest lists. Keyword search, BM25, finds rare exact words like a price or a hostname. Hybrid search uses both. A reranker can reorder that short list. On this run the reorder did not beat the mix, so it was not kept.

The answer step quotes the best sentence and names the page. If nothing matches, it says it does not know. That is why faithfulness is 1.0: the words in the answer were copied from the passage. A copied sentence can still be the wrong sentence. The baseline quote of 25 days is faithful and wrong.

The tuner runs the same questions through seven setups. The composite score is one weighted number. A setup replaces the current one only when that number is strictly higher. Fetching more passages looked thorough and lost, because precision fell to 0.2197.

Harbor Kitchen is the proof that this is not an Acme program. Nine pages, twelve questions written automatically from those pages, then `autorag tune --source examples/harbor-kitchen`. The simple setup scored 0.96. Nothing beat it, so it stayed. The burger question returns $16 from the current menu, not $12 from the 2023 page. The 2023 page never says "status: superseded". The 2025 page says `supersedes: menu-2023`, and that is enough.

## Why is this hard?

The failure is quiet. You still get a fluent sentence. It is just the old policy, or a table row whose column header was cut off. More passages feel safer and make the right page a smaller share of what came back. A score that only checks "did we use the documents' words" will call the wrong quote perfect. You need a known answer, including the number, or you cannot tell.

## How do you know it works?

There is a file of 623 questions with the right answer and the page it lives on. Thirty-seven of them are disagreements between an old page and a new page. Those went from 0.5482 to 1.0. Table questions went from 0.2528 to 0.9710. Questions the handbook cannot answer were refused every time. I would not hide the weak spot: questions that need two pages only moved from 0.5814 to 0.5920, because quoting one sentence is a bad way to join two policies.

The questions were generated from the same sentences as the pages. Acme is synthetic. That makes the test easier than a real intranet where the answer is only implied. Harbor Kitchen's twelve questions were generated the same way, which is why a dumb setup already scored 0.96.

An earlier tune, before a small wording change in the scorer, went from 0.9018 to 0.9474 with the same winning setup. After the change the run is 0.8968 to 0.9435. I re-ran it rather than keep the old table.

## How does it adapt to a new company?

I do not edit code. I point `--source` at their folder, or I change the one `source:` line in `autorag.yaml`. Then `autorag tune` and `autorag serve`. If they have no test file, tune writes starter questions from the current pages and says so. OpenAI or Ollama are a line in that same file if they want a model to phrase the answer. The default does not call them.

## What would you do next?

1. Have someone who has not seen the source sentences write fifty questions, and publish that score next to this one.
2. Teach the quoter to use two sentences when the question clearly needs two facts. That is the flat part of the score.
3. Require a real margin before a setup replaces the current one, so a 0.001 bump does not flip the winner.
4. Put a login in front of the API before it sees a real company's files.

## Score table

| Setup | Score | Right answers | Kept |
| --- | ---: | ---: | --- |
| baseline | 0.8968 | 0.8580 | starting point |
| recency | 0.8952 | 0.8774 | no |
| structure | 0.9420 | 0.9762 | yes |
| hybrid | 0.9435 | 0.9801 | yes, best |
| rerank | 0.9427 | 0.9772 | no |
| wider-chunks | 0.9427 | 0.9772 | no |
| high-k | 0.9297 | 0.9772 | no |
