# Talking through AutoRAG

This is the version Sam can say out loud. The numbers are the ones in `data/acme/tune_results.json`.

## The problem, in one breath

A company has a pile of documents. People ask questions the documents already answer, and they get a wrong number because an old page is still in the folder. I wanted a program that reads the folder, answers with the sentence it used, and then tries a few different ways of searching until it can show that the new way is actually better. Acme is fake, so anyone can check the answers.

## What I decided, and what I gave up

**Search with both meanings and exact words.** A vector search is good when someone says “vacation” and the page says “PTO.” It is weaker on a hostname or a dollar amount. Keyword search (BM25) is the opposite. Hybrid search merges the two ranked lists. On this handbook it was a small gain, 0.9467 to 0.9474, and I kept it because it won, not because the idea sounds good.

**Punish old pages.** The 2023 PTO policy is still in the folder, marked superseded. Without a penalty, a question like “PTO after 3 years” can land on the old wording. The penalty multiplies that page’s score down. It is not a substitute for deleting the file. It is a way to keep the history and still prefer the current page. Turning the penalty on by itself, without fixing how pages are cut, did not win. That run scored 0.8997, under the baseline 0.9018, mostly because context precision fell apart. It was not promoted.

**Cut tables on purpose.** The baseline chops every document into 80-word windows. A markdown table loses its header, so “Chicago” and “$220” are just numbers in a row. The structure chunker keeps a heading with its paragraphs and keeps a whole table together. Table-question correctness went from 0.2822 to 0.9765. That is the clearest win in the run.

**Quote the page instead of asking a chat model to write the answer.** The default generator copies the sentence or the table row that matched. Faithfulness is 1.0 on every config, because the answer’s words are taken from the passage. That number hides something. A quoted sentence can be the wrong sentence and still be “faithful.” The baseline quote for the PTO question is faithful and wrong: it says 25 days, which is the five-year line. Correctness is what catches that. Quoting also will not write a smooth answer that combines two policies in your own words. Multi-document questions are the weak spot for that reason: correctness only moved from 0.7084 to 0.7190. If a real chat model is available, the same passages can be handed to Ollama or OpenAI. I did not use one for the measured run, because the project has to finish with no API key.

**Free local models.** Embeddings in the measured run are MiniLM, a small sentence-transformer, on CPU. The demo container does not ship PyTorch. It uses the same MiniLM exported to ONNX, and the passage vectors are already saved, so the server only embeds the question. A 512 MB free instance cannot hold the PyTorch stack. The tradeoff is two builds of the same model. I checked that the ONNX index still says 22 days, $75, and refuses pet insurance. The score table is from the PyTorch run, not from the ONNX server.

**Keep a config only when the score goes up.** Seven setups. The first is the weak baseline and becomes the current champion. Each later one is saved. It replaces the champion only if the composite is strictly higher. The lexical reranker scored 0.9470 against the hybrid 0.9474, so it was not promoted. Asking for eight passages instead of five kept every right document and crushed precision to 0.2196, so that one was not promoted either. “More context” is not free.

## What the loop is actually measuring

Each question has a known answer and the document ids that contain it. Some questions are supposed to be refused.

- **Correctness.** Did the reply contain the gold wording, and, when the gold answer has numbers, those numbers and not a pile of others? A refusal on a real question scores 0. A refusal on an unanswerable question scores 1.
- **Faithfulness.** Are the reply’s words in the passages we retrieved? A refusal scores 1.
- **Context recall.** Of the documents that should have been found, how many were?
- **Context precision.** Of the documents we did retrieve, how many were on that list?
- **Hit rate.** Did at least one of the right documents show up?
- **MRR.** How high was the first right document?

The composite weights those. Correctness is 0.34 of it. Precision is only 0.08, which is why a big precision drop can still lose even when recall is perfect. I would say that out loud so nobody thinks the weights are magic.

## Limitations I would not hide

The handbook and the questions were generated from the same sentences. That is honest about the facts, and it makes the test easier than a messy real intranet where the answer is implied rather than written as a sentence. A folder with no gold file gets synthetic questions, which mostly check “can we quote this passage back.” That is a smoke test, not a product metric.

Faithfulness at 1.0 is a property of the quoter, not proof the system understands the question.

Thirteen multi-document questions is a thin slice. I would not claim the system is good at questions that need two policies.

The superseded-page penalty is a hand-set multiplier, not something the tuner learned.

The ONNX demo and the PyTorch tune are the same model family, not bit-identical runs. I would not quote the 0.9474 number as the score of the free-host process. It is the score of the tune.

Feedback is stored and merged on the next tune. Nothing in the running server re-tunes itself when someone clicks Helpful. A person, or the Monday job, has to run `autorag tune` again.

There is no login. Do not put a real handbook on the public URL without one.

## What I would do next

1. Have a person write fifty questions who has not seen the source sentences, and report that score next to this one.
2. Let the tuner change the recency penalty instead of hard-coding it, and require the gain to clear a margin so a 0.0004 bump does not flip the champion.
3. For multi-document questions, allow two quotes when they carry different numbers, which is the case the quoter still mishandles.
4. Put an auth wall in front of the API before it sees a real company’s files.
