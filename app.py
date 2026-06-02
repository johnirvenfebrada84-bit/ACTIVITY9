import re
import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel, Nmf
from gensim.models.coherencemodel import CoherenceModel

# Download necessary NLTK data (quietly)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# --- Streamlit UI ---
st.title("📝 Topic Modeling Comparison: LDA vs NMF")
st.write("This app compares Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF) on a sample dataset.")

# Dataset
documents = [
    "Artificial intelligence and machine learning are transforming technology.",
    "Deep learning is a subset of machine learning based on neural networks.",
    "The stock market is influenced by global economic trends and financial news.",
    "Investors analyze market data to predict future stock prices and movements.",
    "Natural language processing is a branch of artificial intelligence.",
    "Financial markets fluctuate based on economic indicators and investor sentiment."
]

if st.button("Run Topic Modeling"):
    with st.spinner("Processing text and training models..."):
        # Preprocessing
        stop_words = set(stopwords.words('english'))
        processed_docs = []
        for doc in documents:
            doc = doc.lower()
            doc = re.sub(r'[^\w\s]', '', doc)
            tokens = doc.split()
            filtered_tokens = [word for word in tokens if word not in stop_words]
            processed_docs.append(filtered_tokens)

        dictionary = corpora.Dictionary(processed_docs)
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        # Train Models
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, random_state=100, passes=10, alpha='auto')
        nmf_model = Nmf(corpus=corpus, id2word=dictionary, num_topics=2, random_state=100, passes=10, w_max_iter=300, h_max_iter=100)

        # Coherence
        coherence_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v').get_coherence()
        coherence_nmf = CoherenceModel(model=nmf_model, texts=processed_docs, dictionary=dictionary, coherence='c_v').get_coherence()

        # Perplexity
        lda_log_perplexity = lda_model.log_perplexity(corpus)
        perplexity_lda = f"{lda_log_perplexity:.4f} (Log Perplexity)"
        perplexity_nmf = "N/A"

        # Display Results
        st.subheader("📊 Comparison Table")
        df_results = pd.DataFrame({
            "Algorithm": ["LDA", "NMF"],
            "Coherence": [coherence_lda, coherence_nmf],
            "Perplexity": [perplexity_lda, perplexity_nmf]
        })
        st.dataframe(df_results)

        st.subheader("💡 Topics Discovered")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**LDA Topics**")
            for idx, topic in lda_model.print_topics(-1):
                st.write(f"**Topic {idx}:** {topic}")
                
        with col2:
            st.markdown("**NMF Topics**")
            for idx, topic in nmf_model.print_topics(-1):
                st.write(f"**Topic {idx}:** {topic}")