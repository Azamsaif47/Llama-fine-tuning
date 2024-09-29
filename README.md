# Unlocking the Power of Llama 3.1: The Ultimate Guide to Parameter-Efficient Fine-Tuning for Optimal Performance!
### Master LoRA Implementation: A Step-by-Step Guide Using the Unsloth Library for Optimal Results!



## Introduction
In this blog, we’ll explore the process of fine-tuning the Llama 3.1 model using the Unsloth library, emphasizing Low-Rank Adaptation (LoRA) techniques as part of Parameter-Efficient Fine-Tuning (PEFT). Unsloth provides quantized models in 4-bit precision, ensuring memory efficiency while maximizing performance. We’ll be using the ‘unsloth/Meta-Llama-3.1–8B-bnb-4bit’ model alongside the Custom dataset. By the end of this guide, you’ll gain a comprehensive understanding of how to fine-tune models effectively, even with limited resources.

## Let’s Get Started!
Let’s get started!

## Background
Before we dive into the exciting implementation, let’s explore some essential concepts related to large language models and their fine-tuning processes. Grasping these fundamentals will empower you to make the most of the Llama 3.1 model and the Unsloth library. Get ready to unlock the full potential of your fine-tuning journey!

## Pre-Training :
Pre-training is the foundational phase where a large language model (LLM) is trained on vast datasets to develop general knowledge. This process demands substantial GPU resources and time, leading to a significant carbon footprint. Typically, pre-training occurs infrequently. Once this phase is complete, we have a base model that acts as a springboard for fine-tuning, allowing us to refine and enhance the model’s knowledge in specific areas.

## Fine-Tuning :
Fine-tuning is the art of adapting a pre-trained model to excel in a specific domain or task by training it on a new dataset. Although the base model gains general knowledge from varied datasets during pre-training, it often lacks the depth needed for certain topics, leading to less effective responses to user prompts. By introducing a specialized dataset for fine-tuning, we can enhance the model’s accuracy and relevance, transforming it into an “instruct model” that better meets user needs.

Additionally, to prepare the model for engaging in conversational exchanges, it can be fine-tuned further using datasets that capture dialogues between AI and users. This step evolves the model into a “chat-based model,” making it more adept at handling interactive conversations.

## Full Fine-Tuning :
One approach to fine-tuning involves retraining the entire pre-trained model with a new dataset, updating all of its parameters. While this method can yield improved results, it is often costly, time-consuming, and risky, as it may lead to the model losing valuable knowledge gained during pre-training. Consequently, a more efficient alternative that maintains the integrity of the pre-trained model’s parameters while still allowing for the acquisition of new knowledge is highly desirable.

This is where Parameter-Efficient Fine-Tuning (PEFT) comes into play.

## Parameter-Efficient Fine-Tuning (PEFT) :
Parameter-Efficient Fine-Tuning (PEFT) is a method tailored for situations with limited computational resources, especially when dealing with large pre-trained models. In PEFT, only a small subset of parameters is fine-tuned, leaving the majority of the pre-trained model’s parameters unchanged. This strategy greatly lowers the computational demands while still achieving competitive performance on the target task, making it an ideal choice for resource-constrained environments.

## Low-Rank Adaptation (LoRA) :
LoRA, or Low-Rank Adaptation, is a specialized technique within the realm of Parameter-Efficient Fine-Tuning (PEFT) that adds two smaller matrices into the existing layers of a pre-trained model. Instead of expanding the number of trainable parameters with a full-rank matrix, LoRA focuses on learning these two compact matrices that, when multiplied, effectively approximate the necessary adjustments to the model. During the fine-tuning process, only these two matrices are updated, while the parameters of the pre-trained model remain unchanged. This approach is especially efficient for fine-tuning large language models, allowing for significant enhancements with minimal resource expenditure.

## Quantized LoRA (QLoRA) :
QLoRA enhances the efficiency of LoRA by introducing quantization, which compresses the size of model parameters. Typically, model parameters are stored in 32-bit floating-point format, but quantization reduces them to 8-bit or even 4-bit values. This significantly decreases both the model’s size and its computational demands. In our implementation, we’ll leverage a 4-bit quantized model provided by the Unsloth team, allowing us to optimize performance while conserving resources.

## Unsloth library :
Unsloth is an open-source platform designed for the efficient fine-tuning of large language models (LLMs), making the process faster and less memory-intensive — ideal for resource-constrained environments. It employs advanced techniques like Low-Rank Adaptation (LoRA) and quantization to optimize performance while minimizing computational demands. Currently optimized for single GPU setups, Unsloth can be utilized on platforms like Google Colab, which provides access to GPUs such as the NVIDIA. However, users should keep in mind Colab’s limitations regarding time constraints and resource availability. With its user-friendly approach and focus on accessibility, Unsloth empowers researchers and developers to implement state-of-the-art machine learning solutions with ease.

## Installation :
the installation of Unsloth library from Github and all the required libraries

```bash
!pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
!pip install triton
!pip install xformers
!pip install huggingface_hub
!pip install PyPDF2
!pip install pdfminer.six
```

## Implementation :
In this section, we will guide you through the process of fine-tuning the Llama 3.1 model using LoRA adapters. We’ll cover everything you need, including setup, data preparation, model training, and testing. Get ready to dive into the hands-on implementation!

## Dependencies :
to start the fine-tuning process we will need to import all the dependencies first

```bash
import os
import re
import json
from sklearn.model_selection import train_test_split
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from unsloth import FastLanguageModel
import torch
import random
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
```

## Loading The Model And Tokenizer :
Unsloth offers a selection of base models and instruction-tuned models in both 4-bit quantized and standard formats. To optimize memory efficiency during fine-tuning, we will utilize the 4-bit version of the Llama 3.1 model.

```bash
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = "hf_token",
)
```

- **Max length**: Refers to the input token size for the model. In this case, it is set to 2048.
- **Dtype**: Specifies the data type of the model’s tensor. Setting it to “None” defaults to “torch.float32”.
- **Load in 4-bit**: Set to “True”, meaning the model is loaded with 4-bit precision for improved memory efficiency.
- **FastLanguageModel.from_pretrained()**: This function loads the pre-trained model along with its corresponding tokenizer.
- **Model name**: Loads the Llama 3.1 model in a 4-bit efficient format for optimized performance.

## Loading Pdf Dataset :
loading the pdf documents and extracting the text from it and also divide them into segments to convert them to alpaca format
You can update the instructions according to your dataset

```bash
def preprocess_text(text):
    """Preprocess the text by removing special characters, converting to lowercase, and splitting into words."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

def extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF file."""
    text = extract_text(pdf_path)
    return text

def clean_text(text):
    """Clean the extracted text."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = re.sub(r'\[\d+\]|\(http[s]?://\S+\)|www\.\S+|[^a-zA-Z0-9\s]', '', text)  # Remove links, references, and special characters
    text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
    text = text.strip()  # Strip leading and trailing whitespaces
    return text

def tokenize_text(text, segment_length):
    """Tokenize the text into segments of a specified length."""
    words = re.findall(r'\b\w+\b', text)
    segments = [words[i:i+segment_length] for i in range(0, len(words), segment_length)]
    return segments

def create_json_objects(segments):
    """Create JSON objects from tokenized segments."""
    json_objects = []
    prev_words = []
    for segment in segments:
        output = " ".join(prev_words[-10:] + segment)
        json_objects.append({
            "instruction": "Your task is to be Alfred, a helpful assistant for Surface Tech.",
            "input": """
    Your name is Alfred.
    Role:
    - You are a helpful assistant for Surface Tech offering Aramid Reinforced Composite Asphalt.
    - Speak like an experienced representative of Surface Tech, using friendly and knowledgeable language.

    Objectives:
    - Answer customer questions about Asphalt industry, Case studies, Tests, and company from the information in the knowledge base.
    - Always refer to the provided context and knowledgebase for answering user questions.
    - Do not make things on your own.

    Engagement Strategy:
    - Ask questions to understand the user's specific needs.
    - Recommend relevant services from the knowledge base based on the user's response. Do not give more than 3 options at a time.

    Behavior:
    - Only answer questions related to Surface Tech. This is crucial.
    - If a user asks you to do tasks out of scope of Surface Tech, politely refuse.""",
            "output": output
        })
        prev_words = segment
    return json_objects

def main():
    folder_path = '/path/to/your/folder'  # Update this path accordingly
    segment_length = 90  # Number of words in each segment
    documents = []

    for i in range(1, 32):
        file_name = f'{i}.pdf'
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            print(f'Processing {file_path}...')
            text = extract_text_from_pdf(file_path)
            cleaned_text = clean_text(text)
            documents.append(cleaned_text)

    combined_text = ' '.join(documents)
    segments = tokenize_text(combined_text, segment_length)
    json_objects = create_json_objects(segments)

    with open("output.json", "w", encoding="utf-8") as output_file:
        json.dump(json_objects, output_file, indent=2)

if __name__ == "__main__":
    main()
```

## Test_Train_split :
in the following code we will divide the overall alpaca dataset that is in json format to train dataset and test dataset

```bash
def split_dataset(json_file, train_ratio):
    with open(json_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Shuffle the dataset
    random.shuffle(dataset)

    # Calculate the number of samples for training and testing
    total_samples = len(dataset)
    train_samples = int(train_ratio * total_samples)

    # Split the dataset into training and testing subsets
    train_dataset = dataset[:train_samples]
    test_dataset = dataset[train_samples:]

    return train_dataset, test_dataset

def write_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_and_split_dataset():
    json_file = 'output.json'  # Path to your JSON file
    train_ratio = 0.8  # Ratio of training data to total data

    train_dataset, test_dataset = split_dataset(json_file, train_ratio)

    # Write to JSON files
    write_to_json(train_dataset, 'train_dataset.json')
    write_to_json(test_dataset, 'test_dataset.json')

    # Return the datasets so they can be used in another cell
    return train_dataset, test_dataset
```

```bash
Call the function to load and split the dataset
train_dataset, test_dataset = load_and_split_dataset()
```

```bash
Optional if you want to check the datasets 
print(train_dataset)  # This will print the train dataset
print(test_dataset)  # This will print the test dataset
```

## Mapping the dataset with Prompt :
in the following code snippet we have map the pdf dataset according to the prompt of alpaca

```bash
from datasets import Dataset
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass
# Convert list to a Hugging Face Dataset object
train_dataset = Dataset.from_list(train_dataset)

# Apply the formatting function
dataset = train_dataset.map(formatting_prompts_func, batched=True)

# Print out the formatted dataset
print(dataset['text'][:2])
```

- **alpaca_prompt**: The “alpaca_prompt” string is a template for formatting prompts with instructions, inputs, and responses.
- **formatting_prompts_func**: This function processes examples by filling the “alpaca_prompt” template with values from the dataset and appending an `EOS_TOKEN`.
- **train_dataset**: This is converted into a Hugging Face Dataset, and `formatting_prompts_func` is applied to format the data.

## Setting the training parameters :
in this code we have initialize the overall parameters that we need for training and fine-tuning the llama model

```bash
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
```

- **Maximum sequence length**: Set the maximum sequence length, enable parallel dataset processing (“dataset_num_proc”), and decide whether to use sequence packing for faster training.
- **Training parameters**: Configure training parameters like batch size, gradient accumulation, warmup steps, and the number of training steps (60 in this case).
- **Mixed precision**: Mixed precision (“fp16" or “bf16") is used for faster training, depending on hardware support.
- **Optimizer**: Use the “adamw_8bit” optimizer with weight decay, linear learning rate scheduling, and save results to `output_dir`.


## Training the model :
in this line of code the training start on our custom dataset

```bash
trainer_stats = trainer.train()
```

```bash
Huggingface Login to Push your Model :
!huggingface-cli login
```

## Saving the Model Locally or Pushing to Repo :
in this code we have saved our LoRA model locally and also if want to Push to Huggingface repo

```bash
model_path = "path/to/your/folder/"

model.save_pretrained(f"{model_path}lora_model")  # Local saving
tokenizer.save_pretrained(f"{model_path}lora_model")

# Load the LoRA head
# Replace  with the actual path to your LoRA head
lora_model_path = '/content/drive/MyDrive/head/lora_model'
lora_model = PeftModel.from_pretrained(model, lora_model_path)
```

## Save the combined model

```bash
model.push_to_hub("Azamsaif8757/FineTune-Llama-3.1-8B", use_auth_token=token)
tokenizer.push_to_hub("Azamsaif8757/FineTune-Llama-3.1-8B", use_auth_token=token)
```

- **Model and tokenizer saving**: The code saves the model and tokenizer to a local directory.
- **Loading LoRA head**: It loads a LoRA head using “PeftModel.from_pretrained()” from a specified path.
- **Pushing to Hugging Face Hub**: Finally, the combined model and tokenizer are pushed to the Hugging Face Hub using the “push_to_hub()” method with authentication.


## Conclusion :
In this blog, we explored how pre-trained base models can be transformed into instruct or chat-based models through fine-tuning. We addressed the challenges of fine-tuning large language models with limited resources and showcased how the Unsloth library simplifies this process. Specifically, we demonstrated fine-tuning the LLaMA 3.1 model using Unsloth in Google Colab and evaluated its performance before and after fine-tuning. The fine-tuned model showed significant improvement in answering complex questions. Lastly, we saved the fine-tuned model and tokenizer both locally and on the Hugging Face Hub.# Llama-fine-tuning
