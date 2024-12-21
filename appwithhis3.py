import streamlit as st
import pandas as pd
from transformers import pipeline

# Initialize the transformer-based text classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Function to evaluate whether the text is AI or human written based on structure, vocabulary, tone, etc.
def evaluate_text(text):
    try:
        # The following are evaluation categories based on the prompt
        categories = [
            "AI", "Human"
        ]
        
        # Custom prompt to classify AI vs Human text based on detailed analysis
        prompt = f'''
       Goal:
Identify whether the provided text is AI-generated or human-written by analyzing structure, tone, style, and language patterns.

Instructions:

Structure:

AI content is well-organized with clear, logical progression.
Human content may have varied structures, occasional tangents, or informal phrasing.
Vocabulary:

AI uses formal, consistent vocabulary (e.g., "significant," "innovative").
Human content includes slang, colloquialisms, and personal opinions.
Repetition:

AI often repeats ideas or terms (e.g., “important,” “critical”).
Humans vary language and may repeat for emphasis.
Tone and Style:

AI has a neutral, professional tone.
Human content may have emotional expression or humor.
Complexity:

AI lacks nuanced, subjective reflection.
Humans offer complex thoughts and admit uncertainty.
Grammar:

AI follows perfect grammar but may have awkward phrasing.
Humans use informal phrasing, slang, and contractions.
Sentence Flow:

AI has consistent, logical sentence flow.
Humans may have pauses or shifts based on real-time thinking.
General Knowledge vs. Personal Experience:

AI relies on generalized facts.
Humans include personal stories or opinions.
Special Words & Patterns:

AI content often uses terms like "optimization," "efficiency," and "scalable."
Human content may have more emotional expression and informal tone.
Task:
Evaluate the text and determine if it's AI or human-written based on the patterns above. Provide a confidence level.



        
        Text: {text}
        '''
        
        # Perform zero-shot classification using the Hugging Face model
        result = classifier(prompt, candidate_labels=categories)
        
        # Return classification result
        classification = result['labels'][0]  # AI or Human
        confidence = result['scores'][0]  # Confidence score
        
        return classification, confidence
    except Exception as e:
        st.error(f"Error evaluating text: {e}")
        return "Error", 0.0

# Function to process CSV and calculate scores for each applicant
def process_csv(file):
    try:
        # Load CSV data
        df = pd.read_csv(file)
        
        # Check if the necessary columns (email, a-j) exist
        required_columns = ['email', 'a', 'b', 'c', 'd', 'e']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"CSV must contain a column named '{col}'.")
                return None
        
        # Analyze each answer and calculate scores
        results = []
        for index, row in df.iterrows():
            total_score = 0
            email = row['email']
            
            # Process answers in columns a to j
            for col in required_columns[1:]:  # Skip the 'email' column
                answer = row[col]
                st.write(f"Processing answer for {email} ({col}): {answer}")
                
                # Classify the answer as AI or Human with confidence level
                answer_type, confidence = evaluate_text(answer)
                if answer_type == "AI":
                    total_score += 2  # Higher score for AI-generated answers
                elif answer_type == "Human":
                    total_score += 1  # Lower score for human-written answers
                else:
                    total_score += 0  # If classification failed, no score added
                
                # Display classification and confidence
                st.write(f"Answer Type: {answer_type}, Confidence: {confidence:.2f}")
                
            # Store the results for each applicant
            results.append({"email": email, "total_score": total_score})

        # Return the results as a DataFrame
        return pd.DataFrame(results)
    
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None

# Streamlit app setup
st.set_page_config(page_title="AI vs Human Detection for Job Applicants")

st.header("Job Applicant AI vs Human Text Classification and Scoring")
st.write("Upload a CSV file containing job applicant answers, and the app will analyze the answers to determine if they are AI-generated or human-written, with scores.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    with st.spinner("Processing the CSV file..."):
        result_df = process_csv(uploaded_file)
        if result_df is not None:
            st.success("Analysis complete!")
            st.subheader("Results:")
            st.dataframe(result_df)

            # Option to download results as a new CSV file
            csv_download = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv_download,
                file_name="applicant_scores.csv",
                mime="text/csv"
            )
