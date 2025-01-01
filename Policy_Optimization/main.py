#The main script will be located here
import os
import openai
from rouge_score import rouge_scorer
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

# Authentication for OpenAI and Google API
openai.api_key = os.getenv('OPENAI_API_KEY')
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# Utility to calculate ROUGE scores
def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

# Generate summary using OpenAI model
def generate_summary(prompt, model="gpt-4", max_tokens=200):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a summarization assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    return response['choices'][0]['message']['content'].strip()

# Write results to Google Sheets
def write_to_google_sheet(spreadsheet_id, range_name, values):
    creds = Credentials.from_authorized_user_file('credentials.json', SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    body = {'values': values}
    result = service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=range_name,
        valueInputOption="RAW",
        body=body
    ).execute()
    print(f"{result.get('updatedCells')} cells updated.")

# Main function to process texts and calculate metrics
def process_and_evaluate(texts, reference_texts, spreadsheet_id, sheet_range):
    results = [["Original Text", "Reference Text", "Generated Summary", "ROUGE-1", "ROUGE-2", "ROUGE-L"]]

    for original, reference in zip(texts, reference_texts):
        try:
            summary = generate_summary(original)
            rouge_scores = calculate_rouge(reference, summary)

            results.append([
                original,
                reference,
                summary,
                rouge_scores['rouge1'],
                rouge_scores['rouge2'],
                rouge_scores['rougeL']
            ])
        except Exception as e:
            print(f"Error processing text: {e}")
            results.append([original, reference, "Error", "Error", "Error", "Error"])

    write_to_google_sheet(spreadsheet_id, sheet_range, results)

# Example usage
if __name__ == "__main__":
    test_texts = [
        "Artificial intelligence is a branch of computer science dealing with the simulation of intelligent behavior in computers.",
        "Machine learning is a method of data analysis that automates analytical model building."
    ]
    reference_summaries = [
        "AI simulates intelligent behavior in computers.",
        "Machine learning automates analytical model building."
    ]

    SPREADSHEET_ID = "your_google_sheet_id_here"
    SHEET_RANGE = "Sheet1!A1:F"

    process_and_evaluate(test_texts, reference_summaries, SPREADSHEET_ID, SHEET_RANGE)