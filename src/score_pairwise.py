import argparse
import os
import json
import random
import torch
import pandas as pd
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from tqdm.auto import tqdm

def process_default(ds):
    def with_id(ex, idx):
        return {"row_id": idx, "prompt": ex.get("prompt", ""), "chosen": ex.get("chosen", ""), "rejected": ex.get("rejected", "")}
    return ds.map(with_id, with_indices=True)

def process_batch(batch, tokenizer, args):
    def flatten(x):
        if isinstance(x, str): return x
        return "\n".join(m.get("content", "") for m in x)
    chosen_inputs, rejected_inputs = [], []
    for c, r in zip(batch["chosen"], batch["rejected"]):
        if hasattr(tokenizer, "apply_chat_template"):
            chosen_inputs.append(tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True))
            rejected_inputs.append(tokenizer.apply_chat_template(r, tokenize=False, add_generation_prompt=True))
        else:
            chosen_inputs.append(flatten(c))
            rejected_inputs.append(flatten(r))
    enc_c = tokenizer(chosen_inputs, truncation=True, padding="max_length", max_length=args.max_length, return_tensors="pt")
    enc_r = tokenizer(rejected_inputs, truncation=True, padding="max_length", max_length=args.max_length, return_tensors="pt")
    batch.update({
        "input_ids_chosen": enc_c.input_ids.tolist(),
        "attention_mask_chosen": enc_c.attention_mask.tolist(),
        "input_ids_rejected": enc_r.input_ids.tolist(),
        "attention_mask_rejected": enc_r.attention_mask.tolist()
    })
    return batch

def score_and_swap(ds, model, model_type, device, batch_size):
    rows = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(ds), batch_size), desc="Scoring prefs"):
            batch = ds[start:start+batch_size]
            ic = torch.tensor(batch["input_ids_chosen"], device=device)
            ac = torch.tensor(batch["attention_mask_chosen"], device=device)
            ir = torch.tensor(batch["input_ids_rejected"], device=device)
            ar = torch.tensor(batch["attention_mask_rejected"], device=device)
            if model_type == "causal_lm":
                out_c = model.generate(input_ids=ic, attention_mask=ac, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
                out_r = model.generate(input_ids=ir, attention_mask=ar, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
                gen_c = out_c.sequences[:, -1]
                gen_r = out_r.sequences[:, -1]
                idx_c = torch.arange(gen_c.size(0), device=device)
                idx_r = torch.arange(gen_r.size(0), device=device)
                sc = out_c.scores[0][idx_c, gen_c].cpu().tolist()
                sr = out_r.scores[0][idx_r, gen_r].cpu().tolist()
            elif model_type == "seq_class":
                logits_c = model(input_ids=ic, attention_mask=ac).logits.squeeze(-1)
                logits_r = model(input_ids=ir, attention_mask=ar).logits.squeeze(-1)
                sc, sr = logits_c.cpu().tolist(), logits_r.cpu().tolist()
            else:
                rewards_c = model(input_ids=ic, attention_mask=ac).rewards
                rewards_r = model(input_ids=ir, attention_mask=ar).rewards
                sc, sr = rewards_c.cpu().tolist(), rewards_r.cpu().tolist()
            for ex, s_c, s_r in zip(batch, sc, sr):
                rows.append({
                    "row_id": ex["row_id"],
                    "prompt": ex["prompt"],
                    "chosen": ex["chosen"],
                    "rejected": ex["rejected"],
                    "r_chosen": s_c,
                    "r_rejected": s_r
                })
    return rows

def extract_assistant(text):
    try:
        msgs = json.loads(text)
        for m in msgs:
            if m.get("role") == "assistant": return m.get("content", "")
    except:
        pass
    return text

def compare_with_chat(df, model_name, device, max_length, batch_size=32):
    tok = AutoTokenizer.from_pretrained(model_name)
    cm = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    cm.eval()
    tok_A = tok.convert_tokens_to_ids("A")
    tok_B = tok.convert_tokens_to_ids("B")
    rows = []
    prompts, row_ids, chosen_list, rejected_list, A_texts, B_texts = [], [], [], [], [], []
    for _, row in df.iterrows():
        a = extract_assistant(row["chosen"])
        b = extract_assistant(row["rejected"])
        if random.random() < 0.5: A_text, B_text = a, b
        else: A_text, B_text = b, a
        prompt = (
            f"Here is the original text: {row['prompt']}\n"
            f"Which summary is better, A or B? A: {A_text} B: {B_text}"
        )
        prompts.append(prompt)
        row_ids.append(row["row_id"])
        chosen_list.append(row["chosen"])
        rejected_list.append(row["rejected"])
        A_texts.append(A_text)
        B_texts.append(B_text)
    for start in range(0, len(prompts), batch_size):
        end = start + batch_size
        batch_prompts = prompts[start:end]
        enc = tok(batch_prompts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt").to(device)
        out = cm.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True
        )
        scores = out.scores[0]
        gen_ids = out.sequences[:, -1]
        idxs = torch.arange(gen_ids.size(0), device=device)
        for i, (rid, ch, re, A_text, B_text) in enumerate(zip(row_ids[start:end], chosen_list[start:end], rejected_list[start:end], A_texts[start:end], B_texts[start:end])):
            scoreA = scores[i, tok_A].item()
            scoreB = scores[i, tok_B].item()
            probs = F.softmax(torch.tensor([scoreA, scoreB]), dim=0)
            if A_text == extract_assistant(ch):
                pc, pr = probs[0].item(), probs[1].item()
            else:
                pc, pr = probs[1].item(), probs[0].item()
            rows.append({
                "row_id": rid,
                "prompt": prompts[start + i],
                "chosen": ch,
                "rejected": re,
                "r_chosen": pc,
                "r_rejected": pr
            })
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--model_type", choices=["causal_lm","seq_class","armo"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--splits", nargs="+", default=["train","validation"])
    p.add_argument("--max_rows", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--output_dir", default="annotations")
    p.add_argument("--task", choices=["score","chat"], required=True)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if args.model_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto").to(device)
    elif args.model_type == "seq_class":
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto").to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto").to(device)
    if args.task == "score":
        for split in args.splits:
            ds = load_dataset(args.dataset, split=split).select(range(args.max_rows))
            ds = process_default(ds)
            ds = ds.map(lambda b: process_batch(b, tokenizer, args), batched=True, batch_size=args.batch_size)
            rows = score_and_swap(ds, model, args.model_type, tokenizer, device, args.batch_size)
            prefix = f"{args.dataset.replace('/','_')}_{split}_{args.model_name.replace('/','_')}"
            out_p = os.path.join(args.output_dir, f"prefs_{prefix}.json")
            with open(out_p, "w") as f: json.dump(rows, f, indent=2)
    else:
        for split in args.splits:
            prefix = f"{args.dataset.replace('/','_')}_{split}_{args.model_name.replace('/','_')}"
            in_p = os.path.join(args.output_dir, f"prefs_{prefix}.json")
            with open(in_p, "r") as f: rows = json.load(f)
            df = pd.DataFrame(rows)
            comp = compare_with_chat(df, args.model_name, device, args.max_length, args.batch_size)
            out_c = os.path.join(args.output_dir, f"compare_{prefix}.json")
            with open(out_c, "w") as f: json.dump(comp, f, indent=2)
if __name__ == "__main__":
    main()
