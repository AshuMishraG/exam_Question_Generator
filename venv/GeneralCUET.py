import csv
import os
import json
import requests
import uuid
from dotenv import load_dotenv
from openai import OpenAI

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

# Directory to save images
IMAGE_DIR = "generated_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Enhanced prompt template for CUET-style General Knowledge questions
def generate_prompt(question_type):
    topics = [
        "Politics: President, State Capitals, Padma Shri Award",
        "Sports: Current Affairs, Rivers, Capitals",
        "History: Currency, GDP",
        "Geography: Current affairs, Discoveries",
        "Economics, Science"
    ]
    topics_str = ", ".join(topics)
    
    return f"""
    You are an AI trained to generate CUET (Common University Entrance Test) General Knowledge sample questions. 
    Focus on the following topics: {topics_str}.
    Create a {question_type} question in the following JSON format:
    {{
        "Question": "Write the question text.",
        "Options": ["option1", "option2", "option3", "option4"],
        "Correct Answer Index": index of correct option, (0-based integer),
        "Explanation": "brief explanation of the correct answer.",
        "Multimedia": "If the question benefits from a multimedia component (image, audio, video), mention its use case and include a brief description of media. If no media is needed, return empty string."
    }}
    Ensure the question is clear and relevant, options are distinct, and the explanation is detailed.
    """

# Function to generate a single question
def generate_question(question_type="multiple-choice"):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in generating CUET-style General Knowledge questions."},
                {"role": "user", "content": generate_prompt(question_type)}
            ],
            temperature=0.7,
            max_tokens=1500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating question: {e}")
        return None

# Function to generate a single image using DALL-E
def generate_image(prompt):
    try:
        detailed_prompt = (
            f"Create an image that resembles a professionally designed General Knowledge exam question paper. "
            f"The image should have a clean white background, black serif fonts, and a formal layout. "
            f"Include text-based questions and options with a structured design, avoiding artistic embellishments. "
            f"Ensure the image is highly readable and formal, similar to an official test paper. "
            f"Related content: {prompt}"
        )

        response = client.images.generate(
            model="dall-e-3",
            prompt=detailed_prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url
        image_name = str(uuid.uuid4()) + ".png"
        local_path = os.path.join(IMAGE_DIR, image_name)
        
        image_response = requests.get(image_url, stream=True)
        image_response.raise_for_status()
        
        with open(local_path, 'wb') as file:
            for chunk in image_response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        return local_path
    except Exception as e:
        print(f"Error generating image: {e}")
        return ""

# Process OpenAI response into CSV format
def process_response(response):
    try:
        # Split the response into individual JSON objects
        json_objects = response.strip().split('\n\n')
        
        questions_data = []
        
        for json_object in json_objects:
            try:
                data = json.loads(json_object)

                question = data.get("Question", "").strip()
                options = "|".join(data.get("Options", [])).strip()
                correct_index = int(data.get("Correct Answer Index", 0))
                explanation = data.get("Explanation", "").strip()
                multimedia = data.get("Multimedia", "").strip()

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

                questions_data.append([
                    question, options, correct_index, "text_only", "text_only",
                    img_url, audio_url, video_url, explanation
                ])
            except json.JSONDecodeError as e:
                print(f"Error processing JSON object: {e}")
                print(f"Raw JSON object: {json_object}")  # Log the raw JSON object for debugging
                continue

        return questions_data
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Main function to generate the CSV file
def generate_cuet_csv(output_file="cuet_gk_questions.csv", target_questions=60):
    questions_data = []

    while len(questions_data) < target_questions:
        print(f"Generating questions... Current count: {len(questions_data)}")
        response = generate_question()
        if response:
            question_data = process_response(response)
            if question_data:
                questions_data.extend(question_data)
                # Trim the list to ensure it doesn't exceed the target
                questions_data = questions_data[:target_questions]
            else:
                print("Skipping due to processing error.")
        else:
            print("Skipping due to an error.")

    # Write to CSV
    try:
        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(questions_data)
        print(f"\nTotal questions saved to {output_file}: {len(questions_data)}")
    except Exception as e:
        print(f"Error saving questions to CSV: {e}")

# Run the script
if __name__ == "__main__":
    generate_cuet_csv(target_questions=60)
