# TechTriage

CS221 Impact Track prototype for helping tech support staff find troubleshooting guidance. Enter a question, predict its category, find related documents, and view source excerpts or an optional AI-generated response.

**Demo only:** The included articles and questions are made up. They are not official Stanford IT advice, and no real staff testing has been done with this starter.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate with `.venv\\Scripts\\activate` instead. Open the local URL printed by Streamlit (usually http://localhost:8501).

Try: `My Xbox connects to wifi but nothing loads`.

## What the app uses

- `classifier.py`: bag-of-words and a linear softmax classifier trained by gradient descent (CS221 Lecture 3).
- `retrieval.py`: BM25 keyword search, optional Sentence-BERT search, and hybrid search.
- `answer.py`: checks for weak matches and drafts a cited answer if an API key is set. Without a key, it shows source excerpts.

To enable AI-generated answers, copy `.env.example` to `.env` and put your OpenAI API key there. Never commit `.env` or enter real student information in this demo.

## Quick check

```bash
python evaluate.py
```

This checks the synthetic training examples only. Its numbers **are not** valid accuracy or user-study results.

## Before real use

Replace the demo data with approved sources, validate the escalation threshold and citations, add reliable handling of private data, and test with actual staff. This prototype is not ready for deployment.

## Research

- Reimers and Gurevych (2019), [Sentence-BERT](https://aclanthology.org/D19-1410/)
- Lewis et al. (2020), [Retrieval-Augmented Generation](https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)

Starter code was created with AI assistance. Document your own use according to the accommodation and course arrangements agreed with your CA.
