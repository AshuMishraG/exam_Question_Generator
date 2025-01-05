from openai import OpenAI
import csv
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the CSV file headers
headers = [
    "Question", "Options", "Correct Answer Index", "Question Type",
    "Option Type", "Question Image URL", "Question Audio URL",
    "Question Video URL", "Explanation"
]

# Function to generate a CUET-style question using OpenAI GPT-4
def generate_cuet_question(topic):
    prompt = f"""
    Create a CUET-style question on the topic '{topic}'.
    Format the output as follows:
    - Question: <Question text>
    - Options: <Option1>|<Option2>|<Option3>|<Option4>
    - Correct Answer Index: <Correct answer index (0-based)>
    - Explanation: <Explanation of the correct answer>
    
    Ensure the question is clear and relevant, options are distinct, and the explanation is detailed.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating question for topic '{topic}': {e}")
        return None

# Function to parse the generated response into structured data
def parse_question(response_text):
    try:
        # Split the response into fields
        lines = response_text.split("\n")
        question = lines[0].split(":")[1].strip()
        options = lines[1].split(":")[1].strip()
        correct_index = int(lines[2].split(":")[1].strip())
        explanation = lines[3].split(":")[1].strip()

        # Define question metadata
        question_type = "text_only"
        option_type = "text_only"
        img_url, audio_url, video_url = "", "", ""

        return [
            question, options, correct_index, question_type, option_type,
            img_url, audio_url, video_url, explanation
        ]
    except Exception as e:
        print(f"Error parsing question: {e}")
        return None

# Function to generate multiple questions and save to CSV
def generate_questions_to_csv(output_csv, topics):
    data = []
    for topic in topics:
        print(f"Generating question for topic: {topic}")
        response = generate_cuet_question(topic)
        if response:
            question_data = parse_question(response)
            if question_data:
                data.append(question_data)
                print(f"Question for '{topic}' added.")
            else:
                print(f"Failed to parse question for '{topic}'.")
        else:
            print(f"Failed to generate question for '{topic}'.")

    # Save data to CSV
    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(data)
        print(f"Questions saved to {output_csv}")
    except Exception as e:
        print(f"Error saving questions to CSV: {e}")

# Main execution
if __name__ == "__main__":
    topics = [
        "Synonyms and Antonyms",
        "Grammar - Error Identification",
        "Reading Comprehension",
        "Sentence Improvement",
        "Passage Analysis"
    ]
    output_csv = "cuet_questions.csv"
    generate_questions_to_csv(output_csv, topics)