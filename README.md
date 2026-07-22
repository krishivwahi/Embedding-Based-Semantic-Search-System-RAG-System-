# Production-Grade Semantic Search & RAG Pipeline

A production level RAG (retrieval-augmented generation) system built from scratch to answer questions and verify scientific claims. (which actually works)

The system uses dense transformer embeddings and FAISS to perform semantic search across the BEIR SciFact dataset (a rigorous academic benchmark containing ~5,000 peer-reviewed biomedical abstracts and expert-annotated queries). An LLM is then grounded on the retrieved context to synthesize answers with exact citations.

## System Architecture

```mermaid
graph LR
    A[Document] --> B[Chunking]
    B --> C[Embeddings]
    C --> D[FAISS]
    D --> E[Retriever]
    E --> F[LLM]
    F --> G[Answer]
```

## Evaluation Results

To ensure that dense embeddings actually retrieve better context than traditional lexical search, I built a baseline TF-IDF retriever and evaluated both approaches on 300 human-labeled test queries from the SciFact benchmark. (I was right btw)

| System     | P@10   | Recall@10 | MRR    |
| ---------- | ------ | --------- | ------ |
| TF-IDF     | 0.0807 | 0.7235    | 0.5500 |
| Embeddings | 0.0893 | 0.7900    | 0.6092 |

As expected, for dense retrieval on scientific literature, embeddings significantly outperformed keyword search, particularly in recall and Mean Reciprocal Rank (placing the correct document higher in the results).

## Running it Locally

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download the dataset:**

   ```bash
   python download_data.py
   ```

3. **Set up the LLM:**
   Create a `.env` file at the root of the project. You can use Groq (that's what I used :\*):

   ```env
   API_KEY = your_key_here
   BASE_URL = url_here
   MODEL= model_name
   ```

4. **Run the Interactive RAG terminal:**
   ```bash
   python ask.py
   ```

## Example Interaction

The system uses strong grounding to prevent hallucination. When asked an overly broad or unanswerable question, the LLM will analyze the retrieved abstracts and explicitly state if a definitive answer isn't present, rather than guessing (which is the complete opposite of what I would do btw):

```text
> can we solve cancer

--- Answer ---
The context does not contain a direct answer to the question "can we solve cancer". The provided sources discuss various aspects of cancer, such as global statistics [1], biological capabilities of cancer cells [2], viral carcinogenesis [3], cancer stem cell hypothesis [4], and cancer chemoprevention [5], but do not provide a definitive answer to whether cancer can be solved.

--- Sources ---
[1] (0.435) 52188256: Global cancer statistics 2018: GLOBOCAN estimates of incidence and mortality wor...
[2] (0.429) 13011249: Hallmarks of Cancer: The Next Generation. The hallmarks of cancer comprise six b...
[3] (0.420) 7451018: Viral Carcinogenesis.. Cancer has been recognized for thousands of years. Egypti...
[4] (0.417) 31902335: The cancer stem cell hypothesis: a guide to potential molecular targets.. Common...
[5] (0.412) 22267225: Cancer chemoprevention: Much has been done, but there is still much to do. State...
```
