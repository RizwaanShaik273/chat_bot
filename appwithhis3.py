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
        Objective:
To evaluate whether the content is AI-generated or human-written, we need to assess various factors based on specific patterns and traits identified in human-generated content.

1. General Characteristics of Human Text:
Purpose-Driven: Human text tends to focus on clear objectives or goals. It includes explanations that are directly related to solving real-world problems or achieving certain outcomes.

Relatable Examples and Details: Human text often uses concrete, practical examples that are easy to relate to, providing specific details to clarify points.

Conciseness with Clarity: Human text is usually concise but comprehensive, focusing on conveying the key idea without unnecessary complexity.

Natural Flow: Human text tends to have a natural flow, with ideas and sentences building logically. It will have a balanced use of short and long sentences for ease of reading.

Tone and Emphasis: The tone often includes personal or emotional elements, like motivation or purpose. It might have slight variations based on context, such as enthusiasm or curiosity.

2. Patterns in AI-Generated Text:
Overly Formal/Neutral Tone: AI-generated text might lean toward overly formal, robotic, or neutral tones without personal emphasis or emotional variation.

Repetitive Ideas: AI content may repeat similar ideas or phrases, reflecting a lack of deep personalization or nuance.

Lack of Practicality: While AI can generate detailed content, it might miss out on specific, practical examples or a relatable context that resonates with the reader.

Complex Sentence Structures: AI-generated text often uses complex or convoluted sentence structures without a clear purpose, which might make it harder to follow the flow.

3. Evaluation Criteria:
Purpose and Specificity: Does the content have a clear goal or purpose? Is it solving a problem or providing a real-world example (Human)?

Relatability and Examples: Does the text include relatable examples and practical details (Human)? Does it feel abstract or generalized (AI)?

Tone and Emotion: Does the text feel naturally conversational or infused with personal motivation? Is it overly neutral or mechanical (AI)?

Flow and Structure: Does the text flow naturally and feel easy to understand with appropriate sentence lengths and transitions (Human)? Is it overly complex or disjointed (AI)?

4. Scoring Methodology:
AI Text: Assign higher scores if the content is neutral, repetitive, lacks practical examples, or has an overly formal tone.
Human Text: Assign lower scores if the content is purpose-driven, includes relatable examples, has natural flow, and carries emotional or personal tone.

        
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
