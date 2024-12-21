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
       Identify if content is AI-generated or human-written by analyzing key traits and patterns.

Human Text Characteristics
Purpose-Driven: Clear goals, problem-solving focus, and actionable insights.
Relatable Examples: Practical, specific details connected to real-world scenarios.
Natural Flow: Logical progression with varied sentence structures.
Personal Tone: Emotional, conversational, and contextually motivated.
AI Text Patterns
Neutral Tone: Robotic or impersonal, lacking emotional depth.
Repetitive Ideas: Recurring phrases or lack of personalization.
Abstract Content: Missing specific, relatable examples.
Complex Sentences: Overly formal or convoluted structures.
Evaluation Criteria
Purpose and Specificity: Does it solve problems with clear goals (Human) or stay abstract (AI)?
Relatability: Are examples practical and detailed (Human) or vague (AI)?
Tone: Is it emotional and personal (Human) or neutral (AI)?
Flow: Does it read naturally (Human) or feel disjointed (AI)?
Scoring
AI: Higher scores for neutral tone, repetition, or lack of examples.
Human: Lower scores for purpose, relatable examples, and emotional tone.

        
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
