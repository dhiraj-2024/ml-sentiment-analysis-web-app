import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
from time import time
from review_analyzer import analyze_sentiment, detect_aspects, detect_emotion

def process_single_review(review, model):
    """Process a single review with all analyses"""
    if pd.isna(review) or str(review).strip() == "":
        return None
        
    sentiment, confidence = analyze_sentiment(review, model)
    aspects = detect_aspects(review)
    emotion = detect_emotion(review)
    
    return {
        "Review": review,
        "Sentiment": sentiment,
        "Confidence": confidence,
        "Aspects": ", ".join(aspects),
        "Emotion": emotion
    }

def analyze_batch_reviews_optimized(df, model, selected_column):
    """Optimized batch review analysis with parallel processing"""
    results = []
    reviews = df[selected_column].tolist()
    total_reviews = len(reviews)
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time()
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Create partial function with model parameter
        process_func = partial(process_single_review, model=model)
        
        # Submit all tasks at once
        futures = {executor.submit(process_func, review): i for i, review in enumerate(reviews)}
        
        # Process completed futures with progress updates
        completed = 0
        for future in tqdm(as_completed(futures), total=total_reviews, desc="Analyzing"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                
                # Update progress every 100 reviews or when complete
                completed += 1
                if completed % 100 == 0 or completed == total_reviews:
                    elapsed = time() - start_time
                    remaining = (elapsed / completed) * (total_reviews - completed)
                    progress = completed / total_reviews
                    
                    # Update progress bar and status
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Analyzing: {completed}/{total_reviews} "
                        f"({progress:.1%}) | "
                        f"Elapsed: {elapsed:.1f}s | "
                        f"Remaining: {remaining:.1f}s | "
                        f"Speed: {completed/elapsed:.1f} reviews/s"
                    )
                    
            except Exception as e:
                st.warning(f"Error processing review: {str(e)}")
                continue
    
    # Final progress update
    progress_bar.progress(1.0)
    status_text.text(f"Analysis complete! Processed {len(results)} reviews in {time()-start_time:.1f} seconds.")
    
    results_df = pd.DataFrame(results)
    
    # Generate summary statistics
    summary = {
        "total_reviews": len(results_df),
        "sentiment_counts": results_df["Sentiment"].value_counts().to_dict(),
        "aspect_counts": results_df["Aspects"].str.split(", ").explode().value_counts().to_dict(),
        "emotion_counts": results_df["Emotion"].value_counts().to_dict()
    }
    
    return results_df, summary

def display_batch_results(results_df, summary):
    """Display batch analysis results with visualizations"""
    st.subheader("Analysis Summary")
    
    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", summary["total_reviews"])
    with col2:
        st.metric("Positive", summary["sentiment_counts"].get("positive", 0))
    with col3:
        st.metric("Neutral", summary["sentiment_counts"].get("neutral", 0))
    with col4:
        st.metric("Negative", summary["sentiment_counts"].get("negative", 0))
    
    # Sentiment Distribution
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=results_df, x="Sentiment", order=["positive", "neutral", "negative"], 
                 palette={"positive": "green", "neutral": "gray", "negative": "red"}, ax=ax)
    ax.set_title("Sentiment Distribution Across Reviews")
    st.pyplot(fig)
    
    # Aspect Distribution
    st.subheader("Aspect Distribution")
    aspect_df = results_df["Aspects"].str.split(", ").explode().value_counts().reset_index()
    aspect_df.columns = ["Aspect", "Count"]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=aspect_df, x="Aspect", y="Count", palette="viridis", ax=ax)
    ax.set_title("Most Discussed Aspects")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    
    # Emotion Distribution
    st.subheader("Emotion Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=results_df, x="Emotion", order=["happy", "neutral", "angry"],
                 palette={"happy": "green", "neutral": "gray", "angry": "red"}, ax=ax)
    ax.set_title("Customer Emotions in Reviews")
    st.pyplot(fig)
    
    # Confidence Distribution
    st.subheader("Confidence Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=results_df, x="Confidence", bins=20, kde=True, ax=ax)
    ax.set_title("Model Confidence Distribution")
    st.pyplot(fig)
    
    # Raw results with pagination
    st.subheader("Sample Results (First 100)")
    st.dataframe(results_df.head(100))
    
    # Download option
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full Results as CSV",
        data=csv,
        file_name="review_analysis_results.csv",
        mime="text/csv"
    )

def batch_analysis_tab(model):
    """Optimized batch analysis tab with all original features"""
    st.header("Batch Review Analysis")
    st.markdown("""
    Upload a CSV or Excel file containing product reviews to analyze them in bulk.
    The file should have at least one column containing review text.
    """)
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], key="batch_upload")
    
    if uploaded_file is not None:
        try:
            # Read file with optimizations
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Show preview
            st.subheader("File Preview")
            st.write(df.head())
            
            # Select column
            review_col = st.selectbox("Select the column containing reviews", df.columns)
            
            if st.button("Analyze Reviews", key="analyze_btn"):
                with st.spinner("Optimized analysis in progress..."):
                    results_df, summary = analyze_batch_reviews_optimized(df, model, review_col)
                    display_batch_results(results_df, summary)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
