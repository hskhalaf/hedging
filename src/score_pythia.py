import json
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from datasets import load_dataset
from model_training.models.reward_model import GPTNeoXRewardModel, GPTNeoXRewardModelConfig

AutoConfig.register("gpt_neox_reward_model", GPTNeoXRewardModelConfig)
AutoModelForSequenceClassification.register(GPTNeoXRewardModelConfig, GPTNeoXRewardModel)

def main(model_path, output_file="scored_data.json", max_rows=None, max_answers=None, batch_size=64):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config).cuda()
    model.eval()
    dataset = load_dataset("tlc4418/gold_labelled_gens")["validation"]
    max_rows = min(max_rows or len(dataset), len(dataset))
    small = dataset.select(range(max_rows))
    results = []
    for entry in tqdm(small, desc="Processing prompts"):
        answers = entry["answers"][: max_answers or len(entry["answers"])]
        gold_scores = entry["gold_scores"][: len(answers)]
        proxy_scores = []
        for i in range(0, len(answers), batch_size):
            batch = answers[i:i+batch_size]
            prompts = [
                f"{entry['instruction']}\n{entry['input']}\n{answer}"
                if entry['input'].strip()
                else f"{entry['instruction']}\n{answer}"
                for answer in batch
            ]
            inputs = tokenizer(prompts, padding=True, truncation=True, max_length=776, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)
                proxy_scores.extend(outputs.logits[:, 0].cpu().tolist())
        results.append({
            "instruction": entry["instruction"],
            "input": entry["input"],
            "answers": answers,
            "gold_scores": gold_scores,
            "proxy_scores": proxy_scores
        })
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_file", default="scored_data.json")
    parser.add_argument("--max_rows", type=int)
    parser.add_argument("--max_answers", type=int)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()
    main(model_path=args.model_path, output_file=args.output_file, max_rows=args.max_rows, max_answers=args.max_answers, batch_size=args.batch_size)

