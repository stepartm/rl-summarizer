import os
from datetime import datetime

import pandas as pd
import torch
from datasets import Dataset
from torch.nn.utils import get_total_norm, clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2LMHeadModel
from tqdm import tqdm

from utils import get_tokenizer


logsoftmax = torch.nn.LogSoftmax(dim=-1)

def compute_loss(model, batch, do_not_need_grad=False):
    input_ids, attention_mask, summary_offset = batch
    max_offset = torch.max(summary_offset)

    with torch.autograd.grad_mode.inference_mode(mode=do_not_need_grad):
        res = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    logits = res.logits[:, -max_offset-1:-1]
    logprobas = logsoftmax(logits)
    labels = input_ids[:, -max_offset:].unsqueeze(2)
    label_logprobas = torch.gather(input=logprobas, dim=2, index=labels).squeeze()

    labels_mask = torch.zeros_like(label_logprobas)

    for i, offset in enumerate(summary_offset):
        for j in range(offset):
            labels_mask[i, - 1 - j] = 1

    loss_per_sequence = - (label_logprobas * labels_mask).sum(dim=-1) / labels_mask.sum(dim=-1)
    return loss_per_sequence.mean()

def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [_["input_ids"] for _ in batch], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left'
    )

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [_["attention_mask"] for _ in batch], batch_first=True, padding_value=0,  padding_side='left'
    )
    summary_offset = torch.tensor([_["summary_offset"] for _ in batch])
    return input_ids, attention_mask, summary_offset

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0")
    BATCH_SIZE = 64
    NUM_WORKERS = 0
    EVAL_EVERY = 1
    LR = 3e-5
    WD = 0
    CLIP_GRAD_VAL = float("inf")

    tokenizer = get_tokenizer()

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(DEVICE)

    sft_train_dataset = Dataset.load_from_disk("datasets/sft/hf_dataset/train").with_format("torch", columns=['input_ids', 'attention_mask', "summary_offset"], device=DEVICE, dtype=torch.int32)
    sft_val_dataset = Dataset.load_from_disk("datasets/sft/hf_dataset/train").with_format("torch", columns=['input_ids', 'attention_mask', "summary_offset"], device=DEVICE, dtype=torch.int32)

    sft_train_dl = torch.utils.data.DataLoader(sft_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    sft_val_dl = torch.utils.data.DataLoader(sft_val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.1)

    iteration_number = 0
    training_step = 0
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    model_name = f"sft_{current_time}_lr={LR}_weight_decay={WD}_clip_grad_val={CLIP_GRAD_VAL}"
    writer = SummaryWriter(log_dir=os.path.join("logs", model_name))

    # while True:
    for _ in range(10):
        model.train()
        for train_batch in tqdm(sft_train_dl, desc=f"Training iteration {iteration_number}"):
            training_step += 1
            training_loss = compute_loss(model=model, batch=train_batch)
            writer.add_scalar(tag="loss/train", scalar_value=training_loss, global_step=training_step)

            training_loss.backward()

            if CLIP_GRAD_VAL < float('inf'):
                grad_norm_before_clipping = clip_grad_norm_(parameters=model.parameters(), error_if_nonfinite=True, max_norm=10)
                writer.add_scalar(tag="grad_norm/before_clipping", scalar_value=grad_norm_before_clipping, global_step=training_step)

            grad_norm_after_clipping = get_total_norm(
                tensors=(p.grad for p in model.parameters() if p.grad is not None), error_if_nonfinite=True
            )
            writer.add_scalar(tag="grad_norm/after_clipping", scalar_value=grad_norm_after_clipping, global_step=training_step)

            optimizer.step()
            optimizer.zero_grad()

        if iteration_number % EVAL_EVERY == 0:
            model.eval()
            validation_loss = 0
            num_validation_batches = 0
            for val_batch in tqdm(sft_val_dl):
                validation_loss += compute_loss(model=model, batch=val_batch, do_not_need_grad=True)
                num_validation_batches += 1

            writer.add_scalar(
                tag="loss/val",
                scalar_value=validation_loss / num_validation_batches,
                global_step=iteration_number
            )

        writer.add_scalar(
            tag="lr",
            scalar_value=scheduler.get_last_lr()[0],
            global_step=iteration_number
        )
        scheduler.step()
        iteration_number += 1

    model.save_pretrained(os.path.join("artifacts", model_name))
