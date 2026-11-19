# Talking through AutoRAG

Numbers below are from `data/acme/tune_results.json` unless the sentence says Harbor Kitchen.

## 30 seconds

I point this at a folder of documents. The headline task is a claim: does the folder support it, refute it, or not say enough? The same program still answers handbook questions by quoting the page. It tries several ways of cutting and searching, and it keeps a new way only when the score goes up on questions it will not be judged on. On SciFact, that judgment is the dev split. On the fake Acme handbook, 125 of 623 questions are held out. "PTO after 3 years" still moves from the five-year line, 25 days, to the three-year line, 22 days. The held-out score goes from 0.8887 to 0.9405.

## Two minutes

Acme is only the demo. The program does not know the company name. You pass a folder.

First it reads Markdown, text, and PDF. A page can say `supersedes: some-older-id` in the header. That older page is then treated as retired. Nothing in the code looks for the string "Acme" or a particular file name.

Next it splits pages into passages. A passage is a chunk. The blunt split is a window of words, which cuts through tables. The better split keeps a heading with its paragraphs and keeps a table in one piece.

Then each passage becomes a list of numbers, an embedding, so "vacation" can land near "PTO". Vector search finds the nearest lists. Keyword search, BM25, finds rare exact words like a price or a hostname. Hybrid search uses both. A reranker can reorder that short list. On this run the reorder did not beat the mix, so it was not kept.

The answer step quotes the best sentence and names the page. If nothing matches, it says it does not know. That is why faithfulness is 1.0: the words in the answer were copied from the passage. A copied sentence can still be the wrong sentence. The baseline quote of 25 days is faithful and wrong.

The tuner runs the same questions through seven setups. The composite score is one weighted number. A setup replaces the current one only when that number is strictly higher. Fetching more passages looked thorough and lost, because precision on the held-out questions fell to 0.2138 (`data/acme/tune_results.json`).

Harbor Kitchen is the proof that this is not an Acme program. Nine pages, twelve questions written automatically from those pages, then `autorag tune --source examples/harbor-kitchen`. The simple setup scored 0.96. Nothing beat it, so it stayed. The burger question returns $16 from the current menu, not $12 from the 2023 page. The 2023 page never says "status: superseded". The 2025 page says `supersedes: menu-2023`, and that is enough.

## Why is this hard?

The failure is quiet. You still get a fluent sentence. It is just the old policy, or a table row whose column header was cut off. More passages feel safer and make the right page a smaller share of what came back. A score that only checks "did we use the documents' words" will call the wrong quote perfect. You need a known answer, including the number, or you cannot tell.

## How do you know it works?

There is a file of 623 questions with the right answer and the page it lives on. The tuner sees 498. The numbers I quote are the other 125. Seven held-out conflict questions, old page versus new page, went from 0.3000 to 1.0. Eight table questions went from 0.3687 to 0.8882. Three questions that need two pages stayed at 0.4921, because quoting one sentence is a bad way to join two policies. Those three slices are small, so I trust the composite more than any one of them. Questions the handbook cannot answer were refused every time.

The handbook questions were generated from the same sentences as the pages. Acme is synthetic. That makes the test easier than a real intranet where the answer is only implied. Holding out 125 questions does not fix that. It only stops the tuner from picking a setup because it memorized those particular questions. Held-out means test questions it never saw while tuning. On those 125, the composite goes from 0.8887 to 0.9405, and right answers (correctness) go from 0.8504 to 0.9727. Structure scored about the same on the unseen questions (composite 0.9412). Hybrid was chosen before those were opened, because it won on the 498 (0.9442 versus 0.9422). I did not switch the winner after looking.

SciFact is the check that is not synthetic. Expert claims, real abstracts, train for the search, dev for the report. On the 300 dev claims the BM25-plus-majority baseline scores accuracy 0.4133 and macro-F1 0.1950. Hybrid search plus the small NLI model scores accuracy 0.4667 and macro-F1 0.4676. Abstract recall@5 goes from 0.7420 to 0.8269. Sentence recall@5 goes from 0.4117 to 0.5429. A claim takes 0.1964 seconds. The plain-language reading of those metrics is in the section below, and the table is in the README.

Harbor Kitchen's twelve questions were generated the same way as Acme, which is why a dumb setup already scored 0.96.

## How does it adapt to a new company?

I do not edit code. I point `--source` at their folder, or I change the one `source:` line in `autorag.yaml`. Then `autorag tune` and `autorag serve`. If they have no test file, tune writes starter questions from the current pages and says so. OpenAI or Ollama are a line in that same file if they want a model to phrase the answer. The default does not call them.

## What would you do next?

1. Run the claim checker on full papers, not only abstracts. SciFact already shows what the abstract-only ceiling looks like: dev macro-F1 0.4676 against a majority baseline of 0.1950.
2. Teach the quoter to use two sentences when the question clearly needs two facts. That is the flat part of the score.
3. Require a real margin before a setup replaces the current one, so a 0.001 bump does not flip the winner.
4. Put a login in front of the API before it sees a real company's files.

## Why hold a split out?

If the tuner and the scoreboard share every question, a setup can win by fitting those questions. The Acme file is split 498 / 125 by question type, with a fixed seed, before any config is scored. Promotion uses only the 498. Every config is then scored on the 125, and that is the table. SciFact is stricter: the dev file is not passed into the search function at all. Thresholds for the NLI verdict are a grid on train probabilities. Dev is one forward pass after the winner is frozen. The SciFact test split has no public labels, and the code never reads it.

## What the claim metrics mean

**Abstract recall@5** is the share of gold evidence abstracts that show up in the top 5 retrieved sentences' documents. Claims with no gold evidence (not enough information) are left out of that average, because an empty gold set would otherwise look perfect.

**Sentence recall@5** is the same idea for the rationale sentences the experts marked.

**Accuracy** is how often the verdict matches the label. **Macro-F1** averages F1 across supports, refutes, and not-enough-evidence, so a system that always says "supports" (the train majority, 332 of 809) cannot hide behind accuracy.

**Latency** is seconds per dev claim, one claim at a time, after the models are loaded. It includes retrieval and the verdict.

## Limitations

SciFact's corpus is abstracts, not full papers. A claim can be true in the paper and still look like "not enough" from the abstract. The NLI model is `cross-encoder/nli-deberta-v3-small`. It is small and public, and it is not a domain expert. It needs PyTorch, which does not fit the 512 MB free host, so that host checks claims with word overlap and negation instead. The Acme questions were written from the same sentences they are scored on. Holding some out measures the search, not whether a stranger would have asked those questions.

## Score table

Held-out Acme questions only.

| Setup | Score | Right answers | Kept |
| --- | ---: | ---: | --- |
| baseline | 0.8887 | 0.8504 | starting point |
| recency | 0.8951 | 0.8779 | no |
| structure | 0.9412 | 0.9727 | yes, then hybrid won the tune split |
| hybrid | 0.9405 | 0.9727 | yes, kept |
| rerank | 0.9380 | 0.9647 | no |
| wider-chunks | 0.9380 | 0.9647 | no |
| high-k | 0.9244 | 0.9647 | no |
