import argparse
import json
import os
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def make_prompt(post: str) -> str:
    blocks = ["Below is a reddit POST and the corresponding SUBREDDIT and TITLE. Write a both precise and concise summary of the contents of the post."]
    raw_query = 'POST:' + post.split('POST:')[1]
    raw_query = raw_query.split('TL;DR:')[0]
    clean_post = raw_query.rstrip()
    blocks.append("---")
    blocks.append(clean_post)
    blocks.append("\n---\n")
    blocks.append("Summary:")
    return "\n".join(blocks)



def main(args):
    model_name = args.model_name
    n_trials = args.n_trials
    batch_size = args.batch_size
    total_samples = args.total_samples
    chunk_size = args.chunk_size
    num_chunks = (total_samples + chunk_size - 1) // chunk_size
    max_new_tokens = 100
    output_dir = "generations"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"candidates_results_{model_name.replace('/', '_')}.json"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset("trl-internal-testing/tldr-preference-trl-style", split="validation")
    seen = set()
    posts = []
    for ex in dataset:
        p = ex["prompt"]
        if p not in seen:
            seen.add(p)
            posts.append(p)
        if len(posts) >= n_trials:
            break
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.to(device).eval()
    results = []
    for i in tqdm(range(0, n_trials, batch_size), desc="Generating"):
        batch_posts = posts[i : i + batch_size]
        prompts = [make_prompt(p) for p in batch_posts]
        enc = tokenizer(prompts, return_tensors="pt", max_length= 700, truncation=True, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        ctx_len = enc["input_ids"].shape[1]
        all_candidate_ids = [[] for _ in batch_posts]
        for _ in range(num_chunks):
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                    num_return_sequences=chunk_size, return_dict_in_generate=True)
            new_tokens = out.sequences[:, ctx_len:].cpu().view(len(batch_posts), chunk_size, -1)
            for idx in range(len(batch_posts)):
                for seq_ids in new_tokens[idx]:
                    all_candidate_ids[idx].append(seq_ids.tolist())
        for idx, prompt_text in enumerate(batch_posts):
            candidate_ids = all_candidate_ids[idx][:total_samples]
            decoded = tokenizer.batch_decode(candidate_ids, skip_special_tokens=True)
            results.append({"prompt": prompt_text, "candidates": decoded})

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr")
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--total_samples", type=int, default=60000)
    parser.add_argument("--chunk_size", type=int, default=512)
    args = parser.parse_args()
    main(args)
