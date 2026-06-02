# Step 1: Import necessary libraries
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel, Nmf
from gensim.models.coherencemodel import CoherenceModel

# Download necessary NLTK data (FIX: Added 'punkt' which is required by CoherenceModel)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ---------------------------------------------------------
# Step 2: Dataset Preparation
# ---------------------------------------------------------

# a. Select a dataset (Dummy dataset with 6 documents)
documents = [
    "Artificial intelligence and machine learning are transforming technology.",
    "Deep learning is a subset of machine learning based on neural networks.",
    "The stock market is influenced by global economic trends and financial news.",
    "Investors analyze market data to predict future stock prices and movements.",
    "Natural language processing is a branch of artificial intelligence.",
    "Financial markets fluctuate based on economic indicators and investor sentiment."
]

# ---------------------------------------------------------
# Step 3: Preprocessing
# ---------------------------------------------------------

def preprocess_text(documents):
    stop_words = set(stopwords.words('english'))
    processed_docs = []
    
    for doc in documents:
        # c. Convert text to lowercase
        doc = doc.lower()
        
        # c. Remove punctuation and special characters using regex
        doc = re.sub(r'[^\w\s]', '', doc)
        
        # c. Tokenize text
        tokens = doc.split()
        
        # c. Remove stopwords
        filtered_tokens = [word for word in tokens if word not in stop_words]
        
        processed_docs.append(filtered_tokens)
        
    return processed_docs

# Perform preprocessing
processed_docs = preprocess_text(documents)

# Create Dictionary and Corpus required for Gensim
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# ---------------------------------------------------------
# Step 4: Select and Train Two Topic Modeling Techniques
# ---------------------------------------------------------

# Technique 1: Latent Dirichlet Allocation (LDA)
print("Training LDA Model...")
lda_model = LdaModel(corpus=corpus,
                     id2word=dictionary,
                     num_topics=2, 
                     random_state=100,
                     passes=10,
                     alpha='auto')

# Technique 2: Non-Negative Matrix Factorization (NMF)
print("Training NMF Model...")
nmf_model = Nmf(corpus=corpus,
                id2word=dictionary,
                num_topics=2,
                random_state=100,
                passes=10,
                w_max_iter=300,
                h_max_iter=100)

# ---------------------------------------------------------
# Step 5: Compare Results (Coherence & Perplexity)
# ---------------------------------------------------------

# 1. Compute Coherence for both models
coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()

coherence_model_nmf = CoherenceModel(model=nmf_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
coherence_nmf = coherence_model_nmf.get_coherence()

# 2. Compute Perplexity
# FIX: Standard practice is to report Log Perplexity. Calculating np.exp() on 
# log_perplexity often causes floating-point underflow (returning 0.0 or an error).
lda_log_perplexity = lda_model.log_perplexity(corpus)
perplexity_lda = f"{lda_log_perplexity:.4f} (Log Perplexity)"
perplexity_nmf = "N/A (Not a probabilistic model)"

# Create the Comparison Table
results_data = {
    "Algorithm (Model)": ["LDA", "NMF"],
    "Coherence": [coherence_lda, coherence_nmf],
    "Perplexity": [perplexity_lda, perplexity_nmf]
}

df_results = pd.DataFrame(results_data)

print("\n" + "="*50)
print("Step 5: Comparison Table")
print("="*50)
print(df_results.to_string(index=False))

# Display the topics found for context
print("\n--- Topics found by LDA ---")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}")

print("\n--- Topics found by NMF ---")
for idx, topic in nmf_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}")

# ---------------------------------------------------------
# Step 6: Discussion
# ---------------------------------------------------------
print("\n" + "="*50)
print("Step 6: Discussion of Results")
print("="*50)

print(f"""
1. Coherence Score:
   - LDA Coherence: {coherence_lda:.4f}
   - NMF Coherence: {coherence_nmf:.4f}
   - Coherence measures how interpretable the topics are to humans.
   - NMF often produces topics with very clear, distinct parts (additive combinations), which can sometimes result in higher coherence scores on smaller or cleaner datasets compared to LDA's probabilistic mixtures.

2. Perplexity Score:
   - LDA Perplexity: {perplexity_lda}
   - NMF Perplexity: {perplexity_nmf}
   - Perplexity is a metric exclusive to probabilistic models (like LDA). It measures how "surprised" the model is by the data. A lower log perplexity indicates a better probabilistic fit.
   - NMF is based on Linear Algebra (factorizing a matrix into two non-negative matrices). It minimizes the error between the original data and the reconstructed data (Frobenius norm), rather than calculating probability. Hence, it does not have a perplexity score.

3. Conclusion:
   - LDA is generally preferred when you need a full statistical model and want to calculate the likelihood of unseen documents (Perplexity).
   - NMF is often faster and can produce more "parts-based" representations (e.g., a topic is defined purely by the presence of specific words without complex probability distributions).
   - Based on the table, if NMF has a higher Coherence score, it suggests that for this specific dataset, NMF extracted more meaningful topics than LDA.
""")