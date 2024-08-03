# py libs
import tiktoken
import math
import time
import os
from contextlib import nullcontext

# torch libs
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# my own files
import model
import prepare_data

# ignore warnings (optional)
import warnings
warnings.filterwarnings("ignore")

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if torch.cuda.is_available() else nullcontext()
use_amp = torch.cuda.is_available()
# scalar = torch.cuda.amp.GradScaler(enabled=use_amp)
scalar = torch.cuda.amp.GradScaler(enabled=use_amp)

# ddp setting here
backend = 'nccl'
use_ddp = int(os.environ.get('RANK', -1)) != -1
grad_accum_steps = 10 * 4

# ddp init here
if use_ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ.get('RANK'))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
    ddp_world_size = int(os.environ.get('WORLD_SIZE'))
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    assert grad_accum_steps % ddp_world_size == 0
    grad_accum_steps = grad_accum_steps / ddp_world_size
else:
    master_process = True
    ddp_world_size = 1




def get_device():
    if torch.cuda.is_available():
        return "cuda"
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     return "mps"
    return "cpu"

class TrainConfig:
    def __init__(self):
        # basic config
        self.batch_size = 4
        self.epochs = 1000
        self.device = torch.device(get_device())
        self.grad_accum_steps = 10 * 4
        
        # checkpoint config
        self.checkpoint = True
        self.checkpoint_interval = 10
        self.checkpoint_dir = "checkpoints"
        self.log_interval = 20
        
        # warm up config, can refer to gpt3 paper, but didn't implement here
        self.linear_step = 2000
        self.max_step = 5000 # 5000 for shakespeare, 600000 for webtext
        
        # optimizer config, all from gpt3 paper
        self.lr = 6e-4
        self.eps = 1e-8
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.weight_decay = 0.1
        self.grad_clip = 1.0
        
        # gpu config
        self.num_workers = 4
        self.pin_memory = True
        self.max_length = 1024

def warmup_lr(step, config):
    lr = config.lr
    linear_step = config.linear_step
    max_step = config.max_step
    if step < linear_step:
        return lr * step / linear_step
    if step > max_step:
        # according to gpt3 paper, 10% here
        return 0.1 * lr
    decay_ratio = (step - linear_step) / (max_step - linear_step)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (1 - coeff) * 0.1 * lr + coeff * lr
    
@torch.no_grad()
def evaluate(model, data_loader, config):
    model.eval()
    tot_loss = 0
    for i, (data, target) in enumerate(data_loader):
        data, target = data.to(config.device), target.to(config.device)
        response, loss = model(data, target)
        tot_loss += loss.detach()
    return tot_loss / len(data_loader)
    
if __name__ == '__main__':
    model_config = model.GPTConfig()
    train_config = TrainConfig()
    train_config.grad_accum_steps = grad_accum_steps

    device = train_config.device
    gpt = model.GPT(model_config)
    gpt.to(device)
    train_data, test_data = prepare_data.get_data()
    n = len(train_data)
    train_data = train_data[int(ddp_rank / ddp_world_size) * n: (int(ddp_rank / ddp_world_size) + 1) * n]
    train_dataloader = DataLoader(prepare_data.MyDataset(train_data, train_config), batch_size=train_config.batch_size, shuffle=True)
    test_dataloader  = DataLoader(prepare_data.MyDataset(test_data, train_config), batch_size=train_config.batch_size, shuffle=True)
    optimizer = gpt.configure_optimizers(train_config)
    
    # import code; code.interact(local=locals())
    # variables inited here
    best_eval_loss = 1e9
    tot_loss = 0
    step = 0
    
    
    # print("compiling the model............ (takes a ~minute)")
    # gpt = torch.compile(gpt)
    # print("model compilation success!")
    
    
    start_time = time.time()
    # train loop below
    for epoch in range(train_config.epochs):
        for i, (train_data, target) in enumerate(train_dataloader):
            # warm up process here!
            # step won't be 0 here
            lr = warmup_lr(step + 1, train_config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            optimizer.zero_grad()
            train_data, target = train_data.to(device), target.to(device)
            gpt.train()
            if i % train_config.grad_accum_steps == 0 and i != 0 and use_ddp:
                model.require_backward_grad_sync = True
            else:
                model.require_backward_grad_sync = False
                
            
            with ctx:
                response, loss = gpt(train_data, target)
            loss = loss / train_config.grad_accum_steps
            tot_loss += loss.detach()
            scalar.scale(loss).backward()
            # mini batch gradient accumulation here
            if i % train_config.grad_accum_steps != 0 or i == 0:
                continue
            
            scalar.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), train_config.grad_clip)
            scalar.step(optimizer)
            scalar.update()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            gap_time = end_time - start_time
            start_time = end_time
            
            
            step += 1
            # if i % train_config.log_interval == 0:
            # if i % 2 == 0:
            if master_process:
                print(f"Epoch: {epoch} | Iteration: {i} | Learning Rate: {lr:.8f} | Loss: {tot_loss / step: .5f} | Time: {gap_time:.4f}s | Tokens per sec: {train_config.batch_size * train_config.grad_accum_steps * ddp_world_size* len(train_data)/gap_time:4f}tokens")
            tot_loss = 0
            if train_config.checkpoint and step % train_config.checkpoint_interval == 0:
                # torch.save(gpt.state_dict(), f"{train_config.checkpoint_dir}/checkpoint_{epoch}_{i}.pt")
                checkpoint_dir = train_config.checkpoint_dir
                save_dir = os.path.join(os.getcwd(), checkpoint_dir)
                eval_loss = evaluate(gpt, test_dataloader, train_config)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    checkpoint = {
                        "model": gpt.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "loss": best_eval_loss,
                        "model_config": model_config,
                        "train_config": train_config,
                        "epoch": epoch,
                        "iter": i,
                        "step": step
                    }
                    print("Best model found! Saving..........")
                    print(f"Step: {step} | Loss: {best_eval_loss}")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(checkpoint, os.path.join(save_dir, f"best_model.pt"))
                    print("Best model saved!")
    if use_ddp:
        destroy_process_group()