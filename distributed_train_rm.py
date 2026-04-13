import os
from datetime import datetime
from typing import Optional

import torch
from datasets import load_from_disk
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import get_total_norm, clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from tqdm import tqdm


def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)


def collate_wrapper(tokenizer, device):
    def collate_(batch):
        winner_inputs = tokenizer(
            [_['winner'] for _ in batch], padding=True, truncation=True,
            return_tensors="pt", add_special_tokens=True, padding_side="left",
        )

        for k, v in winner_inputs.items():
            winner_inputs[k] = v.to(device)

        loser_inputs = tokenizer(
            [_['loser'] for _ in batch], padding=True, truncation=True,
            return_tensors="pt", add_special_tokens=True, padding_side="left",
        )

        for k, v in loser_inputs.items():
            loser_inputs[k] = v.to(device)

        return winner_inputs, loser_inputs

    return collate_


class Trainer:

    def __init__(
            self,
            rank: int,
            n_iterations: Optional[int],
            lr: float,
            weight_decay: float,
            milestones: list,
            gamma: float,
            dataloader_hparams: dict,
            clip_grad_val: float=float('inf'),
            max_norm: float=10,
            eval_every: int = 1
        ):

        self.logsigmoid = torch.nn.LogSigmoid()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
        self.tokenizer.add_eos_token = True

        self.n_iterations = n_iterations
        self.rank = rank
        reward_model = GPT2ForSequenceClassification \
                .from_pretrained('/home/logsumexp/workspace/rl-summarizer/artifacts/sft/Mar31_07-34-36_lr=3e-05_weight_decay=0_clip_grad_val=inf') \
                .to(rank)

        print(f"Model is assigned to rank {rank}")
        reward_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.reward_model = DDP(reward_model, device_ids=[rank])

        self.optimizer = torch.optim.Adam(
            self.reward_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

        comparison_dataset_train = load_from_disk("datasets/comparison/train_preprocessed")
        comparison_dataset_val = load_from_disk("datasets/comparison/val_preprocessed")

        collate_fn = collate_wrapper(tokenizer=self.tokenizer, device=self.rank)

        self.rm_train_dl = torch.utils.data.DataLoader(
            comparison_dataset_train, collate_fn=collate_fn,
            sampler=DistributedSampler(comparison_dataset_train),
            **dataloader_hparams
        )
        self.rm_val_dl = torch.utils.data.DataLoader(
            comparison_dataset_val, collate_fn=collate_fn,
            sampler=DistributedSampler(comparison_dataset_val),
            **dataloader_hparams
        )

        self.training_step = 0

        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.model_name = f"rm/{current_time}_lr={lr}_weight_decay={weight_decay}_clip_grad_val={clip_grad_val}"
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=os.path.join("logs", self.model_name))
        else:
            self.writer = None

        self.clip_grad_val = clip_grad_val
        self.max_norm = max_norm
        self.eval_every = eval_every

    def compute_loss(self, batch, do_not_need_grad=False):
        sample_winner_inputs, sample_loser_inputs = batch

        with torch.autograd.grad_mode.inference_mode(mode=do_not_need_grad):
            winner_logits = self.reward_model(**sample_winner_inputs).logits[:, 1]
            loser_logits  = self.reward_model(**sample_loser_inputs).logits[:, 1]

        return -self.logsigmoid(winner_logits - loser_logits).mean()


    def train_iteration(self, iteration_number: int):
        self.reward_model.train()
        self.rm_train_dl.sampler.set_epoch(iteration_number)

        for train_batch in tqdm(self.rm_train_dl, desc=f"Training iteration {iteration_number}", disable=self.rank != 0):
            self.training_step += 1
            cur_training_step = self.training_step
            training_loss = self.compute_loss(batch=train_batch)

            if self.rank == 0 and self.writer is not None:
                self.writer.add_scalar(tag="loss/train", scalar_value=training_loss, global_step=cur_training_step)

            training_loss.backward()

            if self.clip_grad_val < float('inf'):
                grad_norm_before_clipping = clip_grad_norm_(
                    parameters=self.reward_model.parameters(),
                    error_if_nonfinite=True, max_norm=self.max_norm
                )
                if self.rank == 0 and self.writer is not None:
                    self.writer.add_scalar(
                        tag="grad_norm/before_clipping",
                        scalar_value=grad_norm_before_clipping,
                        global_step=cur_training_step
                    )

            grad_norm_after_clipping = get_total_norm(
                tensors=(p.grad for p in self.reward_model.parameters() if p.grad is not None),
                error_if_nonfinite=True
            )
            if self.rank == 0 and self.writer is not None:
                self.writer.add_scalar(
                    tag="grad_norm/after_clipping",
                    scalar_value=grad_norm_after_clipping,
                    global_step=cur_training_step
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

        if iteration_number % self.eval_every == 0:
            self.reward_model.eval()
            validation_loss = 0
            num_validation_batches = 0
            self.rm_val_dl.sampler.set_epoch(iteration_number)

            for val_batch in tqdm(self.rm_val_dl, disable=self.rank != 0):
                validation_loss += self.compute_loss(batch=val_batch, do_not_need_grad=True)
                num_validation_batches += 1

            if self.rank == 0 and self.writer is not None:
                self.writer.add_scalar(
                    tag="loss/val",
                    scalar_value=validation_loss / num_validation_batches,
                    global_step=iteration_number
                )

        if self.rank == 0 and self.writer is not None:
            self.writer.add_scalar(
                tag="lr",
                scalar_value=self.scheduler.get_last_lr()[0],
                global_step=iteration_number
            )
        self.scheduler.step()

    def train(self):
        if isinstance(self.n_iterations, int):
            for iteration_number in range(self.n_iterations):
                self.train_iteration(iteration_number=iteration_number)
        else:
            iteration_number = 0
            while True:
                self.train_iteration(iteration_number=iteration_number)
                iteration_number += 1

        if self.rank == 0:
            self.reward_model.module.save_pretrained(
                os.path.join("artifacts", self.model_name)
            )

def main(
        rank: int, world_size: int, n_iterations: int | None, lr: float, weight_decay: float,
        milestones: list, gamma: float, batch_size: int, num_workers: int, clip_grad_val: float,
        max_norm: float, eval_every: int
    ):
    ddp_setup(rank=rank, world_size=world_size)
    trainer = Trainer(
        rank=rank,
        n_iterations=n_iterations,
        lr=lr,
        weight_decay=weight_decay,
        milestones=milestones,
        gamma=gamma,
        dataloader_hparams={
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers
        },
        clip_grad_val=clip_grad_val,
        max_norm=max_norm,
        eval_every=eval_every
    )
    trainer.train()
    destroy_process_group()

if __name__ == "__main__":

    n_iterations = 20
    lr = 3e-6
    weight_decay = 0.01
    milestones=[2]
    gamma=0.1
    batch_size = 48
    num_workers = 0
    clip_grad_val = float("inf")
    max_norm = 10
    eval_every = 1

    world_size = torch.cuda.device_count()
    mp.spawn(
        main, nprocs=world_size,
        args=(
            world_size, n_iterations, lr, weight_decay,
            milestones, gamma, batch_size, num_workers, clip_grad_val,
            max_norm, eval_every
        )
    )
