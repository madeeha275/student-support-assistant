# Student Support and Course Feedback Assistant
AI for Everyone – Project 01 · Team: Madeeha, Hadia Arooj

**Live demo:** https://student-support-assistant-fjefqbjc6vxmeuhuwhdb9a.streamlit.app

A five-tab Streamlit app for a teacher reviewing a class after the second assessment period. It estimates whether a
record may need academic support, shows a learning profile group, reads one handwritten feedback digit, and
summarises anonymous English course comments. All models were trained once in Google Colab; the app only loads them.

> The support flag (`needs_support` = 1 when G3 < 10) is a **classroom modelling target**, not a decision about
> promotion, admission, punishment or a real student. A teacher must review every suggestion.

## The five tabs
| Tab | Model | Saved file |
|---|---|---|
| Class overview & support prediction | DecisionTreeClassifier (max_depth 4) + OneHotEncoder | `models/supervised.joblib` |
| Learning profiles | StandardScaler + KMeans (3 groups) on G1, G2, studytime, failures | `models/clusters.joblib`, `models/cluster_descriptions.json` |
| Feedback digit | Small Keras CNN, digits 0–9 (MNIST subset) | `models/images.keras`, `models/image_meta.json` |
| Course comments | NLTK VADER (offline lexicon) + keyword topics (lesson, practice, facilities) | `nltk_data/`, `data/topic_keywords.json` |
| Teaching summary | Offline template (default) or optional Groq Free Plan | – |

Model inputs: G1, G2, studytime, failures, schoolsup, famsup, internet. `record_id`, `split`, the target and G3 are never inputs.

## Results (test rows, measured once)
- Test examples: label 0 = 40, label 1 = 20
- Decision tree (max_depth 4): accuracy 0.917, macro F1 0.907, class 1 precision 0.857, class 1 recall 0.9 (5 test errors)
- Most-frequent baseline: accuracy 0.667, macro F1 0.4, class 1 recall 0.0
- CNN (3 epochs): accuracy 0.86, macro F1 0.861
- VADER vs supplied labels: accuracy 0.967, macro F1 0.967

## Run it on your own computer
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Use the Python version listed in `runtime_versions.txt`. The app opens at http://localhost:8501.

### Optional Groq summary
Create `.streamlit/secrets.toml` (never commit it):
```toml
GROQ_API_KEY = "your-free-groq-key"
GROQ_MODEL = "openai/gpt-oss-20b"
```
Without a key the app works fully in **Offline summary** mode.

## Repository contents
```
app.py                  Streamlit interface + validation
requirements.txt        exact library versions used for training
runtime_versions.txt    Python version used in Colab
models/                 saved models + small JSON metadata
data/                   main.csv, feedback.csv, example_inputs.csv, meta.json, topic_keywords.json, prompt_test_cases.json
sample_images/          one example digit per class
nltk_data/              offline VADER lexicon
source_notices/         data source attribution (keep with the data)
```
The full `images.npz` training arrays and API keys are intentionally **not** in this repository.

## Limitations
- G1 and G2 are allowed because they are known after the second period; G3 is the final grade the target was made
  from, so it is excluded. A prediction before the course began could not use G1 or G2 and would be much weaker.
- KMeans groups are learning profiles, not class labels or permanent labels about ability; group numbers are arbitrary.
- The CNN was trained on MNIST digits; real student handwriting or scanned forms may be misread. The teacher confirms
  the digit, and it is never used as an academic mark.
- VADER is an English lexicon: sarcasm, negation and mixed opinions can fail; Urdu / Roman Urdu are not supported.
- Groq free quotas may run out; the app then falls back to the Offline summary.

## Data sources
See `source_notices/`. Tabular data: UCI Student Performance (mathematics course, selected fields).
Images: MNIST. Comments are authored classroom examples, not real student responses.
