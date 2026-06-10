# LLM Prompting Techniques: Tree of Thought & Chain of Thought

This repository contains two Jupyter notebooks demonstrating advanced LLM prompting strategies applied to real machine learning and NLP tasks.

---

## Notebooks

### 1. Choosing Right Parameters Using Tree of Thought
**File:** `Choosing_right_parameters_using_Tree_of_Thought_.ipynb`

Uses the **Tree of Thought (ToT)** prompting technique to intelligently select the best hyperparameters for a Random Forest classifier.

**What it does:**
- Loads the Breast Cancer dataset from scikit-learn
- Runs a full hyperparameter grid search across `n_estimators`, `max_depth`, and `min_samples_split`
- Records CV accuracy, train/test accuracy, overfitting gap, and training time for each combination
- Sends all results to an LLM with a structured Tree of Thought prompt that reasons across four branches:
  - **Branch A** – Performance Analysis (best CV accuracy)
  - **Branch B** – Overfitting Analysis (train vs test gap)
  - **Branch C** – Training Cost Analysis (time vs gain)
  - **Branch D** – Bias-Variance Tradeoff (depth analysis)
- The LLM merges all branches and recommends the single best configuration with a reasoned explanation
- Saves results to `hyperparameter_results.csv` and the LLM analysis to `tree_of_thought_analysis.txt`

**Key concepts:** Tree of Thought prompting, hyperparameter tuning, Random Forest, overfitting analysis

---

### 2. Sentiment Analysis with Chain of Thought from Customer Reviews
**File:** `Sentiment_with_CoT_from_Customer_Reviews.ipynb`

Uses **Few-Shot Chain of Thought (CoT)** prompting to perform nuanced sentiment analysis on customer reviews.

**What it does:**
- Takes a dataset of 10 customer product reviews
- Builds a detailed few-shot CoT prompt with 3 worked examples (Positive, Neutral, Negative)
- For each review, the LLM follows a 5-step reasoning chain:
  1. Identify positive-sentiment phrases
  2. Identify negative-sentiment phrases
  3. Check for contradictions or mixed sentiment
  4. Reason step-by-step
  5. Output a final label: **Positive / Neutral / Negative**
- Parses the structured LLM output to extract the final label programmatically
- Outputs a clean DataFrame with each review and its predicted sentiment

**Key concepts:** Chain of Thought prompting, few-shot learning, sentiment analysis, structured output parsing

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| scikit-learn | ML model and dataset |
| pandas / numpy | Data handling |
| Groq API | LLM inference |
| Google Colab | Notebook environment |

---

## Setup

1. Clone this repository
2. Open the notebooks in Google Colab (or Jupyter)
3. Add your Groq API key to Colab secrets as `groq`
4. Run all cells top to bottom

```python
# The notebooks read the key like this:
from google.colab import userdata
os.environ['GROQ_API_KEY'] = userdata.get('groq')
```

---

## Project Structure

```
├── Choosing_right_parameters_using_Tree_of_Thought_.ipynb
├── Sentiment_with_CoT_from_Customer_Reviews.ipynb
└── README.md
```

---

## Key Takeaways

- **Tree of Thought** is powerful for multi-criteria decision making — instead of asking the LLM one question, you guide it to reason from multiple angles before converging on an answer.
- **Chain of Thought** with few-shot examples dramatically improves structured output quality for classification tasks like sentiment analysis.
- Both techniques make LLM reasoning more transparent and reliable compared to direct prompting.
