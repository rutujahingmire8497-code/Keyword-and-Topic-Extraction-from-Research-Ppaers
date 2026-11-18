# Text Summarization of News and Research Papers

Course project (team of 2): Rutuja Hingmire (EN25552010), Vaishali Dhayagude (EN25552004)

Compares **extractive** summarization (TextRank, baseline) with **abstractive** summarization
(BART / T5, advanced model) using **ROUGE**, and ships a small summarizer tool.

## Datasets (public)
- CNN/DailyMail 3.0.0 — news article → highlights
- arXiv summarization — paper text → abstract

All text is cleaned with `src/preprocess.py`, which removes emails, URLs, phone numbers and @handles.
Cite the dataset papers and Hugging Face pages in the final report.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the comparison
```bash
python run_experiments.py --dataset cnn_dailymail --n 100
python run_experiments.py --dataset arxiv --n 50
```
Scores (ROUGE-1/2/L F1, runtime) are saved to `results/`.
Start with `--n 20` to test quickly; use a GPU (e.g. Google Colab) for larger runs.

## Run the demo tool
```bash
streamlit run app.py
```

## Project structure
```
src/preprocess.py    cleaning, PII removal, sentence splitting
src/textrank.py      baseline: TF-IDF + PageRank extractive summarizer
src/abstractive.py   advanced: BART / T5 summarizer
src/evaluate.py      ROUGE-1, ROUGE-2, ROUGE-L
src/data.py          dataset loading
run_experiments.py   runs all models, writes results/
app.py               Streamlit summarizer tool
```

## Results
Fill in after running (put the table from `results/` here):

| Dataset | Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Time (s) |
|---|---|---|---|---|---|
| CNN/DailyMail | TextRank | | | | |
| CNN/DailyMail | BART | | | | |
| CNN/DailyMail | T5 | | | | |
| arXiv | TextRank | | | | |
| arXiv | BART | | | | |
| arXiv | T5 | | | | |

## References
- Mihalcea & Tarau (2004), TextRank: Bringing Order into Texts
- Lewis et al. (2020), BART
- Raffel et al. (2020), T5
- Lin (2004), ROUGE
- See, Liu & Manning (2017), CNN/DailyMail summarization
- Cohan et al. (2018), arXiv/PubMed summarization dataset
