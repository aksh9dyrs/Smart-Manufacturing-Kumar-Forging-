from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    context_precision
)
from datasets import Dataset
import pandas as pd
from typing import List, Dict
import psycopg2
from datetime import datetime

def get_db_connection():
    return psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5132"
    )

def prepare_evaluation_data(conn, num_samples=100):
    """Prepare data for Ragas evaluation from the database"""
    query = """
        SELECT id, event_type, machine_name, notes, embedding, timestamp
        FROM manufacturing_events
        WHERE embedding IS NOT NULL
        LIMIT 10
    """
    df = pd.read_sql(query, conn, params=(num_samples,))
    
    # Create synthetic questions and ground truth answers
    questions = []
    ground_truths = []
    contexts = []
    
    for _, row in df.iterrows():
        # Create different types of questions
        questions.extend([
            f"Tell me about the {row['event_type']} event on machine {row['machine_name']}",
            f"What happened during the {row['event_type']} event on {row['machine_name']}?",
            f"Describe the {row['event_type']} event that occurred on {row['machine_name']}"
        ])
        
        # Create corresponding ground truths
        ground_truths.extend([
            f"Event {row['id']} was a {row['event_type']} event on machine {row['machine_name']}. {row['notes']}",
            f"The {row['event_type']} event on machine {row['machine_name']} occurred at {row['timestamp']}. {row['notes']}",
            f"On {row['machine_name']}, there was a {row['event_type']} event. {row['notes']}"
        ])
        
        # Create contexts (using the event details as context)
        contexts.extend([
            f"Event ID: {row['id']}\nType: {row['event_type']}\nMachine: {row['machine_name']}\nNotes: {row['notes']}\nTimestamp: {row['timestamp']}",
            f"Event ID: {row['id']}\nType: {row['event_type']}\nMachine: {row['machine_name']}\nNotes: {row['notes']}\nTimestamp: {row['timestamp']}",
            f"Event ID: {row['id']}\nType: {row['event_type']}\nMachine: {row['machine_name']}\nNotes: {row['notes']}\nTimestamp: {row['timestamp']}"
        ])
    
    return Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "contexts": contexts
    })

def evaluate_rag_system(conn, model_responses: List[str], evaluation_data: Dataset):
    """Evaluate the RAG system using Ragas metrics"""
    # Add model responses to the evaluation data
    evaluation_data = evaluation_data.add_column("answer", model_responses)
    
    # Define metrics to evaluate
    metrics = [
        faithfulness,
        answer_relevancy,
        context_relevancy,
        context_recall,
        context_precision
    ]
    
    # Run evaluation
    result = evaluate(
        evaluation_data,
        metrics=metrics
    )
    
    return result

def generate_model_responses(conn, questions: List[str]) -> List[str]:
    """Generate responses using the existing RAG system"""
    responses = []
    for question in questions:
        # Get question embedding
        query = """
            SELECT id, event_type, machine_name, notes, embedding
            FROM manufacturing_events
            WHERE embedding IS NOT NULL
            LIMIT 5
        """
        # Use the existing get_question_embedding function from app2.py
        from app2 import get_question_embedding
        embedding = get_question_embedding(question)
        
        if embedding:
            df = pd.read_sql(query, conn, params=(embedding,))
            context = "\n".join([
                f"Event {row['id']}: {row['event_type']} on machine {row['machine_name']}. {row['notes']}"
                for _, row in df.iterrows()
            ])
            
            # Use the existing cached_generate_content function from app2.py
            from app2 import cached_generate_content
            prompt = f"""Based on the following manufacturing data, answer this question: {question}

            Context:
            {context}

            Please provide a direct answer using the actual data shown above."""
            
            response = cached_generate_content(prompt)
            responses.append(response)
        else:
            responses.append("Could not generate embedding for the question.")
    
    return responses

def main():
    conn = get_db_connection()
    
    # Prepare evaluation data
    print("Preparing evaluation data...")
    evaluation_data = prepare_evaluation_data(conn)
    
    # Generate model responses
    print("Generating model responses...")
    model_responses = generate_model_responses(conn, evaluation_data["question"])
    
    # Run evaluation
    print("Running Ragas evaluation...")
    results = evaluate_rag_system(conn, model_responses, evaluation_data)
    
    # Print results
    print("\nEvaluation Results:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    
    conn.close()

if __name__ == "__main__":
    main() 