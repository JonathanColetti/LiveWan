"""Encode the prompt bank with umt5-xxl once, on GPU, into a single blob.

The T5 encoder is 11 GB and takes ~40 s per prompt on CPU, which is why the
original code cached per prompt. Here it runs once on the GPU for the whole
bank and is then never loaded again -- data generation and both training stages
read embeddings straight out of `prompts.pt`.
"""
import os, sys, argparse
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'wan21_repo'))
sys.path.insert(0, HERE)

from wan.configs import WAN_CONFIGS
from wanstreamer.prompts import prompt_bank

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default=os.path.join(HERE, 'checkpoints/wan21_13b'))
ap.add_argument('--out', default=os.path.join(HERE, 'data/prompts.pt'))
ap.add_argument('--text-len', type=int, default=512)
args = ap.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
cfg = WAN_CONFIGS['t2v-1.3B']
dev = torch.device('cuda:0')

from wan.modules.t5 import T5EncoderModel
t5 = T5EncoderModel(text_len=args.text_len, dtype=cfg.t5_dtype, device=dev,
                    checkpoint_path=f'{args.ckpt}/{cfg.t5_checkpoint}',
                    tokenizer_path=f'{args.ckpt}/google/umt5-xxl')

prompts = prompt_bank()
print(f'encoding {len(prompts)} prompts + 1 negative on GPU')


def enc(batch):
    outs = t5(batch, dev)
    padded = []
    for x in outs:
        x = x.float().cpu()
        if x.shape[0] < args.text_len:
            x = torch.cat([x, torch.zeros(args.text_len - x.shape[0], x.shape[1])], 0)
        padded.append(x[:args.text_len])
    return torch.stack(padded)


chunks = []
B = 8
for i in range(0, len(prompts), B):
    chunks.append(enc(prompts[i:i + B]).to(torch.float16))
    print(f'  {min(i+B, len(prompts))}/{len(prompts)}')
pos = torch.cat(chunks)
neg = enc([cfg.sample_neg_prompt]).to(torch.float16)

torch.save({'prompts': prompts, 'pos': pos, 'neg': neg,
            'neg_prompt': cfg.sample_neg_prompt}, args.out)
print(f'wrote {args.out}  pos {tuple(pos.shape)}  neg {tuple(neg.shape)}')
