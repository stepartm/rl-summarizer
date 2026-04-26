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
from transformers import GPT2LMHeadModel
from tqdm import tqdm

from utils import get_tokenizer


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


def collate_wrapper(tokenizer):
    def collate_fn(batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [_["input_ids"] for _ in batch], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left'
        )

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [_["attention_mask"] for _ in batch], batch_first=True, padding_value=0,  padding_side='left'
        )
        summary_offset = torch.tensor([_["summary_offset"] for _ in batch])
        return input_ids, attention_mask, summary_offset

    return collate_fn


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
            max_norm: float=float('inf'),
            eval_every: int = 1
        ):

        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.tokenizer = get_tokenizer()

        self.n_iterations = n_iterations
        self.rank = rank
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.to(rank)

        print(f"Model is assigned to rank {rank}")
        model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = DDP(model, device_ids=[rank])

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

        dataset_train = (
            load_from_disk("datasets/sft/hf_dataset/train")
            .with_format(
                "torch",
                columns=['input_ids', 'attention_mask', "summary_offset"],
                device=rank,
                dtype=torch.int32
            )
        )
        dataset_val = (
            load_from_disk("datasets/sft/hf_dataset/train")
            .with_format(
                "torch",
                columns=['input_ids', 'attention_mask', "summary_offset"],
                device=rank,
                dtype=torch.int32
            )
        )

        collate_fn = collate_wrapper(tokenizer=self.tokenizer)

        self.train_dl = torch.utils.data.DataLoader(
            dataset_train, collate_fn=collate_fn,
            sampler=DistributedSampler(dataset_train),
            **dataloader_hparams
        )
        self.val_dl = torch.utils.data.DataLoader(
            dataset_val, collate_fn=collate_fn,
            sampler=DistributedSampler(dataset_val),
            **dataloader_hparams
        )

        self.training_step = 0

        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.model_name = f"sft/{current_time}_lr={lr}_weight_decay={weight_decay}_clip_grad_val={max_norm}"
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=os.path.join("logs", self.model_name))
        else:
            self.writer = None

        self.max_norm = max_norm
        self.eval_every = eval_every

    def compute_loss(self, batch, do_not_need_grad=False):
        input_ids, attention_mask, summary_offset = batch
        max_offset = torch.max(summary_offset)

        with torch.autograd.grad_mode.inference_mode(mode=do_not_need_grad):
            res = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        logits = res.logits[:, -max_offset-1:-1]
        logprobas = self.logsoftmax(logits)
        labels = input_ids[:, -max_offset:].unsqueeze(2)
        label_logprobas = torch.gather(input=logprobas, dim=2, index=labels).squeeze()

        labels_mask = torch.zeros_like(label_logprobas)

        for i, offset in enumerate(summary_offset):
            for j in range(offset):
                labels_mask[i, - 1 - j] = 1

        loss_per_sequence = - (label_logprobas * labels_mask).sum(dim=-1) / labels_mask.sum(dim=-1)
        return loss_per_sequence.mean()


    def train_iteration(self, iteration_number: int):
        self.model.train()
        self.train_dl.sampler.set_epoch(iteration_number)

        for train_batch in tqdm(self.train_dl, desc=f"Training iteration {iteration_number}", disable=self.rank != 0):
            self.training_step += 1
            cur_training_step = self.training_step
            training_loss = self.compute_loss(batch=train_batch)

            if self.rank == 0 and self.writer is not None:
                self.writer.add_scalar(tag="loss/train", scalar_value=training_loss, global_step=cur_training_step)

            training_loss.backward()

            if self.max_norm < float('inf'):
                grad_norm_before_clipping = clip_grad_norm_(
                    parameters=self.model.parameters(),
                    error_if_nonfinite=True, max_norm=self.max_norm
                )
                if self.rank == 0 and self.writer is not None:
                    self.writer.add_scalar(
                        tag="grad_norm/before_clipping",
                        scalar_value=grad_norm_before_clipping,
                        global_step=cur_training_step
                    )

            grad_norm_after_clipping = get_total_norm(
                tensors=(p.grad for p in self.model.parameters() if p.grad is not None),
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
            self.model.eval()
            validation_loss = 0
            num_validation_batches = 0
            self.val_dl.sampler.set_epoch(iteration_number)

            for val_batch in tqdm(self.val_dl, disable=self.rank != 0):
                validation_loss += self.compute_loss(batch=val_batch, do_not_need_grad=True)
                num_validation_batches += 1

            if self.rank == 0 and self.writer is not None:
                self.writer.add_scalar(
                    tag="loss/val",
                    scalar_value=validation_loss / num_validation_batches,
                    global_step=iteration_number
                )

                self.model.module.save_pretrained(
                    os.path.join("artifacts", self.model_name, f"epoch_{iteration_number}")
                )
                torch.save(
                    self.optimizer.state_dict(),
                    os.path.join("artifacts", self.model_name, f"epoch_{iteration_number}", "optimizer_state")
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
            self.model.module.save_pretrained(
                os.path.join("artifacts", self.model_name, "final")
            )
            torch.save(
                self.optimizer.state_dict(),
                os.path.join("artifacts", self.model_name, "final", "optimizer_state")
            )

def main(
        rank: int, world_size: int, n_iterations: int | None, lr: float, weight_decay: float,
        milestones: list, gamma: float, batch_size: int, num_workers: int,
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
        max_norm=max_norm,
        eval_every=eval_every
    )
    trainer.train()
    destroy_process_group()

if __name__ == "__main__":

    n_iterations = None
    lr = 3e-5
    weight_decay = 0
    milestones = [5]
    gamma = 0.1
    batch_size = 64
    num_workers = 0
    max_norm = 10
    eval_every = 2

    world_size = torch.cuda.device_count()
    mp.spawn(
        main, nprocs=world_size,
        args=(
            world_size, n_iterations, lr, weight_decay,
            milestones, gamma, batch_size, num_workers,
            max_norm, eval_every
        )
    )
