import transformers
from openai import OpenAI, AzureOpenAI
import json
import os
from PIL import Image
from tqdm import tqdm
import wandb
import random
from PIL import Image
from tqdm import tqdm
import argparse
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Sequence, List

@dataclass
class GeneratorArgument:
    seed: int = field(default=42)
    version: Optional[str] = field(default="v0")
    log: bool = field(default=False)
    print: bool = field(default=False)
    project_name: Optional[str] = field(default="Skin_LLaVA_Convgen")
    model_name: str = field(default="gpt-4-1106-preview")
    num_samples: Union[int, None] = field(default=None) # If using all samples, set to None
    num_example: int = field(default = 2)
    root_dir: str = field(default="/data2/ArtLab_LLM/label/train_231109/JPEGImages/") # Directory of JSON annotation to generate data with
    system_path: str = field(default="./prompt/system_message.txt")
    example_path: str = field(default="./prompt/examples/")
    output_path: str = field(default="./output/output.json")

def generate_query_imgdir(sample, root_dir):
    with open(root_dir+sample, 'r') as file:
        data = json.load(file)
    img_dir = root_dir+data["file_name"]

    keys_to_drop = ["caption", "part", "file_name", "rosacea", "acne", "eczema"]
    for key in keys_to_drop:
        data.pop(key, None)

    data["dryness"] = data.pop("hydration", None)

    return {"query":str(data), "img_name":img_dir}

def generate_conversation(system_message, samples, query, model_name):

    azure = ["gpt-35-turbo", "gpt-35-turbo-16k", "gpt-35-turbo-instruct", "gpt-4", "gpt-4-32k", "gpt-4-1106-preview", "gpt-4-vision-preview"]
    # public = ["gpt-3.5-turbo-1106", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-instruct", "gpt-4"]

    if model_name in azure:
        client = AzureOpenAI(
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key = os.getenv("AZURE_OPENAI_KEY"),
            api_version = "2023-08-01-preview"
        )
    # elif model_name in public:
    #     client = OpenAI(
    #         api_key = os.getenv("PUBLIC_OPENAI_KEY"),
    #     )
    else:
        raise ValueError("Model name is unrecognizable.")
    
    messages = [
        {"role": "system", "content": system_message}
    ]
    for sample in samples:
        messages.append({"role": "user", "content": sample['context']})
        messages.append({"role": "assistant", "content": sample['response']})
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model = model_name,
        messages = messages
    )

    content = response.choices[0].message.content
    completion_tokens = response.usage.completion_tokens
    prompt_tokens = response.usage.prompt_tokens

    if model_name in ["gpt-35-turbo", "gpt-3.5-turbo-1106", "gpt-35-turbo", "gpt-35-turbo-instruct"]:
        coef = (0.000002,0.0000015) #completion, prompt
    elif model_name in ["gpt-4"]:
        coef = (0.00006,0.00003)
    elif model_name in ["gpt-4-1106-preview", "gpt-4-vision-preview"]:
        coef = (0.00003,0.00001)
    elif model_name in ["gpt-4-32k", "gpt-4-32k"]:
        coef = (0.00012, 0.00006)
    elif model_name in ["gpt-35-turbo-16k", "gpt-3.5-turbo-16k"]:
        coef = (0.000004, 0.000003)
    else:
        raise ValueError("Unknown model name")
    
    price = completion_tokens*coef[0] + prompt_tokens*coef[1]

    return content, price

def preprocess_example(file):
    context = file.split("\n")[0]
    response = file[len(context):].strip()
    return {"context": context, "response": response}


if __name__ == "__main__":

    parser = transformers.HfArgumentParser(GeneratorArgument)
    CFG = parser.parse_args_into_dataclasses()[0]

    # Set OpenAI API Key and seed
    os.environ['AZURE_OPENAI_KEY'] = '7f9fc78bc5c44ad9a9f46cff6e1f6562'
    os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://coresearch.openai.azure.com/'
    random.seed(CFG.seed)

    # Select image-paired annotation JSON files to generate conversations with
    if CFG.num_samples:
        test_samples =  random.sample([f for f in os.listdir(CFG.root_dir) if f.endswith(".json")], CFG.num_samples)
        test_container = [generate_query_imgdir(sample, CFG.root_dir) for sample in test_samples]
    else:
        test_samples = [f for f in os.listdir(CFG.root_dir) if f.endswith(".json")]
        test_container = [generate_query_imgdir(sample, CFG.root_dir) for sample in test_samples]

    # Randomly samples 2-shot examples from example pool
    file_container = []

    with open(CFG.system_path, 'r') as file:
        system_message = file.read()

    example_files = os.listdir(CFG.example_path)
    selected_files = random.sample(example_files, CFG.num_example)

    for file_name in selected_files:
        file_path = os.path.join(CFG.example_path, file_name)
        with open(file_path, 'r') as file:
            file_content = file.read()
            file_container.append(file_content)

    fewshot_samples = [preprocess_example(file) for file in file_container]

    # Generate conversation data
    if CFG.log:
        wandb.init(project=CFG.project_name, group=CFG.model_name + " " + CFG.version + " s." + str(CFG.seed), name=CFG.model_name + " " + str(CFG.num_example) + "-shot " + CFG.version)
        table = wandb.Table(columns=['Image Name', 'Image', 'Query', 'Generated Conversation'])

    sum = 0
    all_data_to_save = []

    for test_sample in tqdm(test_container):
        conv_generated, price = generate_conversation(system_message, fewshot_samples, test_sample['query'], CFG.model_name)
        sum += price
        sample_image = Image.open(test_sample['img_name'])
        img_name = os.path.basename(test_sample['img_name'])

        if CFG.log:
            table.add_data(img_name, wandb.Image(sample_image), test_sample['query'], conv_generated)
            
        if CFG.print:
            print(os.path.basename(test_sample['img_name']))
            sample_image.show()
            print(test_sample['query'])
            print(conv_generated + "\n\n")
        
        data_to_save = {
            'Image': test_sample['img_name'],
            'Query': test_sample['query'],
            'Conv': conv_generated
        }

        all_data_to_save.append(data_to_save)
    
    with open(CFG.output_path, 'w') as json_file:
        json.dump(all_data_to_save, json_file, indent=4)

    if CFG.log:
        wandb.log({"Log": table})
        
        artifact = wandb.Artifact("prompts", type="dataset")
        artifact.add_file(CFG.system_path, "system_message.txt")
        for file_name in os.listdir(CFG.example_path):
            file_path = os.path.join(CFG.example_path, file_name)
            artifact.add_file(file_path)

        wandb.log_artifact(artifact)

    print("Price: $", sum)