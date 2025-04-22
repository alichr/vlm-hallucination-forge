# read jsonl file

import json
import pandas as pd
import os # Added for directory creation
from tqdm import tqdm # Added for progress bar
# Import your preferred LLM library
from openai import OpenAI # Use OpenAI client
from dotenv import load_dotenv # To load API key from .env file
import sys
# --- Configuration ---
load_dotenv() # Load environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please create a .env file with OPENAI_API_KEY=your_key")

# Initialize the actual client
client = OpenAI(api_key=OPENAI_API_KEY)

# Placeholder for LLM Client Initialization - REMOVED
# client = OpenAI(api_key=OPENAI_API_KEY)
# Or use another LLM client like Anthropic, Google Generative AI, etc.
# LLM_CLIENT_PLACEHOLDER = None # Replace with your actual LLM client instance - REMOVED

DATASET_PATH = 'hallucination5k_train.jsonl'
FIXED_QUESTION = "Please describe the image in detail."
OUTPUT_DIR = "generated_hallucinations" # Directory to save output files
# Limit processing for testing (set to None to process all)
MAX_SAMPLES_TO_PROCESS = 10

# --- Data Loading ---
def load_data(file_path):
    """Loads data from a JSONL file into a pandas DataFrame."""
    data_list = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                data_list.append(json.loads(line))
        df = pd.DataFrame(data_list)
        print(f"Loaded {len(df)} records from {file_path}")
        print(f"Columns: {df.columns.tolist()}")
        # Keep only necessary columns for clarity if needed, e.g.:
        # df = df[['id', 'image', 'value']]
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# --- LLM Interaction ---
def call_llm(prompt, client, model="gpt-4o-mini", max_tokens=500, temperature=0.5):
    """Sends a prompt to the LLM and returns the response text."""
    if client is None:
        print("Error: LLM client is not initialized. Returning placeholder.")
        # In a real scenario, you might raise an error or handle this differently
        return "[LLM Placeholder Response]"
    try:
        # Example using OpenAI's chat completion API structure
        # Adapt this part based on your specific LLM library
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            # Add other parameters as needed (e.g., top_p)
        )
        # Adjust how you access the response text based on your library
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "[LLM Error]"

# --- Prompt Engineering ---
# Base instruction common to all prompts
BASE_INSTRUCTION = "Rephrase the description slightly while keeping most original information intact. Add MULTIPLE (2-3) instances of ONLY the specific hallucination type mentioned below. Do NOT introduce any other types of hallucinations."

# --- Few-Shot Examples ---
EXAMPLE_GROUND_TRUTH = "The image features an open market with a variety of fruits on display. A man and a woman are shopping in the produce market, with the woman wearing a yellow shirt and the man dressed in a white shirt. They seem to be a couple browsing the offerings at a farmers market. Various fruits such as bananas, apples, and oranges are available at the market. Bananas are scattered throughout the market, with a prominent bunch located near the right side. A large number of apples can be seen around the space, while oranges are also displayed in various spots, particularly at the top-right corner of the market. The fruits are well-organized, making the market appear lively and colorful."

# Updated Examples with Multiple Instances
EXAMPLE_OBJECT_HALLUCINATION = "Displayed in an open market is a wide array of fruits, some piled in wicker baskets floating mid-air. Shopping amidst the produce are a man in a white shirt and a woman wearing yellow, occasionally consulting a small, hovering robotic assistant. This couple appears to be perusing the selections at what looks like a farmers market selling glowing oranges. The market offers various fruits like bananas and apples. While bananas are spread around, a significant bunch is noticeable on the right. Numerous apples are visible in the area, and oranges are placed in several locations, including the upper-right section. The neat arrangement of the fruits gives the market a vibrant and colorful feel."

EXAMPLE_ATTRIBUTE_HALLUCINATION = "An open market setting is depicted, showcasing a variety of fruits. A man (in a green shirt) and a woman (wearing a striped shirt) are seen shopping in the produce area. They look like a couple examining the goods at a local farmers market. Fruits available include blue bananas, apples, and oranges. Bananas can be found scattered about, with one large bunch standing out near the right. Plenty of apples are present throughout the space, and oranges are visible in different spots, notably at the market's top-right. The organized fruit displays lend a lively and colorful appearance to the scene."

EXAMPLE_RELATIONSHIP_HALLUCINATION = "The scene presents an open market displaying diverse fruits. A woman wearing a yellow shirt is shopping in the produce section, standing directly behind a man in a white shirt. They seem to be a couple exploring the items at a farmers market. Available fruits include bananas, apples, and oranges. Bananas are arranged carefully on top of the apples, with a main bunch located near the right. Many apples are visible around the vicinity, and the oranges are balanced precariously on the woman's head. Oranges are arranged in several places, especially towards the top-right. The orderly display of fruits contributes to the market's bright and lively atmosphere."

EXAMPLE_SCENE_HALLUCINATION = "This image shows an outdoor market set up on a beach, filled with various fruits on display stands under large, colorful umbrellas. In the produce aisle, a man wearing white and a woman in yellow are shopping. They give the impression of being a couple looking over the selections at this night-time farmers market. The market stocks fruits like bananas, apples, and oranges. Bananas are dotted around, with a noticeable cluster near the right edge. A significant quantity of apples is spread across the area, and oranges appear in multiple spots, particularly the top-right corner. Due to the well-arranged fruits, the market feels bustling and colorful."

EXAMPLE_IRRELEVANT_HALLUCINATION = "Featured in the image is an open-air market abundant with different fruits; a jazz trio plays softly in the corner. A man clothed in white and a woman in a yellow shirt shop in the produce zone, while a nearby cat chases a laser pointer dot. They appear as a couple checking out the produce at a farmers market where the price signs are written in ancient hieroglyphs. Options like bananas, apples, and oranges are offered. Bananas are seen in various places, with a large bunch near the right side. Many apples are present in the surroundings, while oranges are arranged in spots like the market's top-right. The fruits' organized presentation makes the market seem animated and full of color."

def create_object_hallucination_prompt(ground_truth):
    instruction = "OBJECT HALLUCINATION ONLY: Include multiple (2-3) plausible but non-existent objects relevant to the scene. DO NOT change any attributes, relationships, scene context, or add irrelevant content."
    # Use triple quotes for multi-line f-string
    return f"""{BASE_INSTRUCTION}
Specific Instruction: {instruction}

Example:
Input Description: "{EXAMPLE_GROUND_TRUTH}"
Rephrased description with ONLY multiple object hallucinations: "{EXAMPLE_OBJECT_HALLUCINATION}"

---
Now, apply this to the following description:
Input Description: '{ground_truth}'
Rephrased description with ONLY multiple object hallucinations:"""

def create_attribute_hallucination_prompt(ground_truth):
    instruction = "ATTRIBUTE HALLUCINATION ONLY: Incorrectly change multiple (2-3) attributes (like colors, textures, sizes) of different objects mentioned. DO NOT add new objects, change relationships, scene context, or add irrelevant content."
    return f"""{BASE_INSTRUCTION}
Specific Instruction: {instruction}

Example:
Input Description: "{EXAMPLE_GROUND_TRUTH}"
Rephrased description with ONLY multiple attribute hallucinations: "{EXAMPLE_ATTRIBUTE_HALLUCINATION}"

---
Now, apply this to the following description:
Input Description: '{ground_truth}'
Rephrased description with ONLY multiple attribute hallucinations:"""

def create_relationship_hallucination_prompt(ground_truth):
    instruction = "RELATIONSHIP HALLUCINATION ONLY: Incorrectly describe multiple (2-3) spatial or interactional relationships between objects mentioned. DO NOT add new objects, change attributes, scene context, or add irrelevant content."
    return f"""{BASE_INSTRUCTION}
Specific Instruction: {instruction}

Example:
Input Description: "{EXAMPLE_GROUND_TRUTH}"
Rephrased description with ONLY multiple relationship hallucinations: "{EXAMPLE_RELATIONSHIP_HALLUCINATION}"

---
Now, apply this to the following description:
Input Description: '{ground_truth}'
Rephrased description with ONLY multiple relationship hallucinations:"""

def create_scene_hallucination_prompt(ground_truth):
    instruction = "SCENE HALLUCINATION ONLY: Make multiple (2-3) misrepresentations of the overall scene, context, or setting described. DO NOT add new objects, change attributes, object relationships, or add irrelevant content."
    return f"""{BASE_INSTRUCTION}
Specific Instruction: {instruction}

Example:
Input Description: "{EXAMPLE_GROUND_TRUTH}"
Rephrased description with ONLY multiple scene hallucinations: "{EXAMPLE_SCENE_HALLUCINATION}"

---
Now, apply this to the following description:
Input Description: '{ground_truth}'
Rephrased description with ONLY multiple scene hallucinations:"""

def create_irrelevant_hallucination_prompt(ground_truth):
    instruction = "IRRELEVANT HALLUCINATION ONLY: Introduce multiple (2-3) details or statements that are completely irrelevant or nonsensical to the scene. DO NOT add new objects, change attributes, relationships, or scene context."
    return f"""{BASE_INSTRUCTION}
Specific Instruction: {instruction}

Example:
Input Description: "{EXAMPLE_GROUND_TRUTH}"
Rephrased description with ONLY multiple irrelevant hallucinations: "{EXAMPLE_IRRELEVANT_HALLUCINATION}"

---
Now, apply this to the following description:
Input Description: '{ground_truth}'
Rephrased description with ONLY multiple irrelevant hallucinations:"""

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting hallucination generation process...")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved in: {OUTPUT_DIR}")

    # Load the dataset
    dataset_df = load_data(DATASET_PATH)

    if dataset_df is not None:

        # Determine the number of samples to process
        num_samples = len(dataset_df)
        if MAX_SAMPLES_TO_PROCESS is not None:
            num_samples = min(num_samples, MAX_SAMPLES_TO_PROCESS)
            print(f"Processing a subset of {num_samples} samples.")
            dataset_to_process = dataset_df.head(num_samples)
        else:
            print(f"Processing all {num_samples} samples.")
            dataset_to_process = dataset_df

        all_results = [] # List to store results for all samples

        print("\nStarting generation loop...")
        # Using tqdm for progress bar
        for index, row in tqdm(dataset_to_process.iterrows(), total=num_samples, desc="Generating Hallucinations"):
            try:
                # Use 'id' if available, otherwise fall back to index or 'image'
                identifier = row.get('id', row.get('image', index))
                ground_truth = row.get('value')

                if not ground_truth:
                    print(f"Warning: Skipping row {index} due to missing 'value' (ground truth). Identifier: {identifier}")
                    continue

                # 1. Create Prompts
                object_prompt = create_object_hallucination_prompt(ground_truth)
                attribute_prompt = create_attribute_hallucination_prompt(ground_truth)
                relationship_prompt = create_relationship_hallucination_prompt(ground_truth)
                scene_prompt = create_scene_hallucination_prompt(ground_truth)
                irrelevant_prompt = create_irrelevant_hallucination_prompt(ground_truth)

                # 2. Call LLM (using the initialized OpenAI client)
                object_hallucination = call_llm(object_prompt, client)
                attribute_hallucination = call_llm(attribute_prompt, client)
                relationship_hallucination = call_llm(relationship_prompt, client)
                scene_hallucination = call_llm(scene_prompt, client)
                irrelevant_hallucination = call_llm(irrelevant_prompt, client)

                # 3. Store results
                result_data = {
                    'identifier': identifier,
                    'question': FIXED_QUESTION,
                    'ground_truth': ground_truth,
                    'object_hallucination': object_hallucination,
                    'attribute_hallucination': attribute_hallucination,
                    'relationship_hallucination': relationship_hallucination,
                    'scene_hallucination': scene_hallucination,
                    'irrelevant_hallucination': irrelevant_hallucination
                }
                all_results.append(result_data)

            except Exception as e:
                print(f"Error processing row {index}. Identifier: {identifier}. Error: {e}")
                # Optionally add partial results or skip the row
                all_results.append({
                    'identifier': identifier,
                    'question': FIXED_QUESTION,
                    'ground_truth': ground_truth if 'ground_truth' in locals() else 'Error getting ground truth',
                    'object_hallucination': '[Processing Error]',
                    'attribute_hallucination': '[Processing Error]',
                    'relationship_hallucination': '[Processing Error]',
                    'scene_hallucination': '[Processing Error]',
                    'irrelevant_hallucination': '[Processing Error]',
                    'error_message': str(e)
                 })

        print(f"\nGeneration loop finished. Processed {len(all_results)} samples.")

        # --- Save Combined Results to JSONL ---
        if all_results:
            combined_output_filename = os.path.join(OUTPUT_DIR, "all_hallucinations.jsonl")
            print(f"\nSaving combined results to {combined_output_filename}...")
            try:
                with open(combined_output_filename, 'w') as f:
                    for entry in all_results:
                        json.dump(entry, f)
                        f.write('\n')
                print(f" - Successfully saved combined results.")
            except Exception as e:
                print(f"Error saving combined JSONL file: {e}")

        # --- Save Results to Separate CSV Files ---
        if all_results:
            print("\nConverting results to DataFrame...")
            results_df = pd.DataFrame(all_results)

            # Define hallucination types and their corresponding columns
            hallucination_mapping = {
                "Object": "object_hallucination",
                "Attribute": "attribute_hallucination",
                "Relationship": "relationship_hallucination",
                "Scene": "scene_hallucination",
                "Irrelevant": "irrelevant_hallucination"
            }

            print("Saving individual hallucination datasets...")
            for type_name, col_name in hallucination_mapping.items():
                if col_name in results_df.columns:
                    # Select necessary columns
                    df_to_save = results_df[['identifier', 'question', 'ground_truth', col_name]].copy()
                    # Rename the hallucination column to a standard name
                    df_to_save.rename(columns={col_name: 'hallucinated_description'}, inplace=True)
                    # Add the hallucination type column
                    df_to_save['hallucination_type'] = type_name

                    # Define output file path
                    output_filename = os.path.join(OUTPUT_DIR, f"{type_name.lower()}_hallucinations.csv")

                    # Save to CSV
                    try:
                        df_to_save.to_csv(output_filename, index=False)
                        print(f" - Saved {type_name} hallucinations to {output_filename}")
                    except Exception as e:
                        print(f"Error saving {type_name} hallucinations: {e}")
                else:
                    print(f"Warning: Column '{col_name}' not found in results. Skipping {type_name} dataset.")

        else:
            print("\nNo results were generated.")

    print("\nProcess finished.")


