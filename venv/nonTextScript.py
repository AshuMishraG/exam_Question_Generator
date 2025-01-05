from openai import OpenAI
import csv
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define CSV headers
headers = [
    "Question", "Options", "Correct Answer Index", "Question Type",
    "Option Type", "Question Image URL", "Question Audio URL",
    "Question Video URL", "Explanation"
]

# Prompt template for CUET-style questions
def generate_prompt(question_type):
    return f"""
    You are an AI trained to generate CUET (Common University Entrance Test) sample questions. 
    Create a {question_type} question in the following JSON format:
    {{
    "Question": "Write the question text.",
    "Options": ["option1", "option2", "option3", "option4"],
    "Correct Answer Index": index of correct option, (0-based integer),
    "Explanation": "brief explanation of the correct answer.",
     "Multimedia": "If the question benefits from a multimedia component (image, audio, video), mention its use case and include a brief description of media. If no media is needed, return empty string"
    }}
    Ensure the question is clear and relevant, options are distinct, and the explanation is detailed.
    """

# Function to generate a single question
def generate_question(question_type="multiple-choice"):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in generating CUET-style sample questions."},
                {"role": "user", "content": generate_prompt(question_type)}
            ],
            temperature=0.7,
            max_tokens=700,  # Increased token limit
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating question: {e}")
        return None


# Function to generate a single image using DALL-E
def generate_image(prompt):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024" # Corrected image size
        )
        return response.data[0].url
    except Exception as e:
        print(f"Error generating image: {e}")
        return ""

# Process OpenAI response into CSV format
def process_response(response):
    try:
      
        data = json.loads(response)

        question = data.get("Question", "").strip()
        options = "|".join(data.get("Options", [])).strip()
        correct_index = int(data.get("Correct Answer Index", 0))
        explanation = data.get("Explanation", "").strip()
        multimedia = data.get("Multimedia","").strip()

        # Initialize media URLs
        img_url, audio_url, video_url = "", "", ""

        # Generate image if needed
        if "image" in multimedia.lower():
          try:
             img_prompt = f"Generate an image related to the following question: {question}, {multimedia}"
             img_url = generate_image(img_prompt)
          except Exception as e:
            print(f"Error processing image generation: {e}")

        # Set audio/video URL placeholders
        audio_url = "https://example.com/audio.mp3" if "audio" in multimedia.lower() else ""
        video_url = "https://example.com/video.mp4" if "video" in multimedia.lower() else ""
        
        return [
            question, options, correct_index, "text_only", "text_only",
            img_url, audio_url, video_url, explanation
        ]
    except Exception as e:
        print(f"Error processing response: {e}")
        return None


# Main function to generate the CSV file
def generate_cuet_csv(output_file="cuet_questions.csv", num_questions=50):
    questions_data = []

    # Generate questions
    for i in range(num_questions):
        print(f"Generating question {i + 1}/{num_questions}...")
        response = generate_question()
        if response:
           question_data = process_response(response)
           if question_data:
                questions_data.append(question_data)
           else:
                print(f"Skipping question {i + 1} due to processing error.")
        else:
            print(f"Skipping question {i + 1} due to an error.")

    # Write to CSV
    try:
        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(questions_data)
        print(f"\nQuestions saved to {output_file}")
    except Exception as e:
        print(f"Error saving questions to CSV: {e}")

# Run the script
if __name__ == "__main__":
    generate_cuet_csv(num_questions=50)