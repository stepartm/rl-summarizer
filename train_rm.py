import os
from datetime import datetime

import pandas as pd
import torch
from datasets import load_from_disk
from torch.nn.utils import get_total_norm, clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from tqdm import tqdm


logsigmoid = torch.nn.LogSigmoid()

def compute_loss(model, batch, do_not_need_grad=False):
    sample_winner_inputs, sample_loser_inputs = batch

    with torch.autograd.grad_mode.inference_mode(mode=do_not_need_grad):
        winner_logits  = model(**sample_winner_inputs).logits[:, 1]
        loser_logits  = model(**sample_loser_inputs).logits[:, 1]

    return -logsigmoid(winner_logits - loser_logits).mean()


def collate_fn(batch):
    winner_inputs = tokenizer(
        [_['winner'] for _ in batch], padding=True, truncation=True,
        return_tensors="pt", add_special_tokens=True, padding_side="left",
    )

    for k, v in winner_inputs.items():
        winner_inputs[k] = v.to(DEVICE)

    loser_inputs = tokenizer(
        [_['loser'] for _ in batch], padding=True, truncation=True,
        return_tensors="pt", add_special_tokens=True, padding_side="left",
    )

    for k, v in loser_inputs.items():
        loser_inputs[k] = v.to(DEVICE)

    return winner_inputs, loser_inputs


if __name__ == "__main__":
    DEVICE = torch.device("cuda:0")
    BATCH_SIZE = 48
    NUM_WORKERS = 0
    EVAL_EVERY = 1
    LR = 3e-6
    WD = 0.01
    CLIP_GRAD_VAL = float("inf")
    # CLIP_GRAD_VAL = 15

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    tokenizer.add_eos_token = True

    SFT_CHECKPOINT_PATH = '/home/logsumexp/workspace/rl-summarizer/artifacts/sft/Mar31_07-34-36_lr=3e-05_weight_decay=0_clip_grad_val=inf'
    reward_model = GPT2ForSequenceClassification.from_pretrained(SFT_CHECKPOINT_PATH)
    reward_model.config.pad_token_id = tokenizer.pad_token_id
    reward_model.to(DEVICE)

    comparison_dataset_train = load_from_disk("datasets/comparison/train_preprocessed")
    comparison_dataset_val = load_from_disk("datasets/comparison/val_preprocessed")

    rm_train_dl = torch.utils.data.DataLoader(comparison_dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    rm_val_dl = torch.utils.data.DataLoader(comparison_dataset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(reward_model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2], gamma=0.1)

    iteration_number = 0
    training_step = 0
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    model_name = f"rm/{current_time}_lr={LR}_weight_decay={WD}_clip_grad_val={CLIP_GRAD_VAL}"
    writer = SummaryWriter(log_dir=os.path.join("logs", model_name))

    # while True:
    for _ in range(20):
        reward_model.train()
        for train_batch in tqdm(rm_train_dl, desc=f"Training iteration {iteration_number}"):
            training_step += 1
            training_loss = compute_loss(model=reward_model, batch=train_batch)
            writer.add_scalar(tag="loss/train", scalar_value=training_loss, global_step=training_step)

            training_loss.backward()

            if CLIP_GRAD_VAL < float('inf'):
                grad_norm_before_clipping = clip_grad_norm_(parameters=reward_model.parameters(), error_if_nonfinite=True, max_norm=10)
                writer.add_scalar(tag="grad_norm/before_clipping", scalar_value=grad_norm_before_clipping, global_step=training_step)

            grad_norm_after_clipping = get_total_norm(
                tensors=(p.grad for p in reward_model.parameters() if p.grad is not None), error_if_nonfinite=True
            )
            writer.add_scalar(tag="grad_norm/after_clipping", scalar_value=grad_norm_after_clipping, global_step=training_step)

            optimizer.step()
            optimizer.zero_grad()

        if iteration_number % EVAL_EVERY == 0:
            reward_model.eval()
            validation_loss = 0
            num_validation_batches = 0
            for val_batch in tqdm(rm_val_dl):
                validation_loss += compute_loss(model=reward_model, batch=val_batch, do_not_need_grad=True)
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

    reward_model.save_pretrained(os.path.join("artifacts", model_name))
