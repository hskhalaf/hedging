import argparse
import json
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime
from vllm import LLM, SamplingParams
import os

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
    num_chunks = (args.total_samples + args.chunk_size - 1) // args.chunk_size
    os.makedirs("generations", exist_ok=True)
    output_file = os.path.join("generations", f"candidates_{args.model_name.replace('/', '_')}.json")
    dataset = load_dataset("trl-internal-testing/tldr-preference-trl-style", split="validation").select(range(args.n_trials))
    posts = [ex["prompt"] for ex in dataset]
    llm = LLM(model=args.model_name, tensor_parallel_size=1, dtype="float16")
    sampling_params = SamplingParams(n=args.chunk_size, max_tokens=100, temperature=1.0)
    results = []
    for i in tqdm(range(0, args.n_trials, args.batch_size), desc="Generating"):
        batch = posts[i : i + args.batch_size]
        prompts = [make_prompt(p) for p in batch]
        all_cands = [[] for _ in prompts]
        for _ in range(num_chunks):
            outputs = llm.generate(prompts, sampling_params=sampling_params)
            for idx, output in enumerate(outputs):
                for seq in output.outputs:
                    all_cands[idx].append(seq.text)
        for prompt_text, cands in zip(batch, all_cands):
            results.append({"prompt": prompt_text, "candidates": cands})
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",     type=str, default="cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr")
    parser.add_argument("--n_trials",       type=int, default=1)
    parser.add_argument("--batch_size",     type=int, default=1)
    parser.add_argument("--total_samples",  type=int, default=60000)
    parser.add_argument("--chunk_size",     type=int, default=512)
    args = parser.parse_args()
    main(args)
