import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

class Config:
    # 硬件优化参数
    batch_size = 2048          # 利用A100的大显存
    mixed_precision = True    # 启用混合精度
    
    # 模型超参数
    max_len = 100             # 长序列建模
    embed_dim = 512           # 大嵌入维度
    num_heads = 16            # 多注意力头
    num_blocks = 6            # 深层Transformer
    dropout = 0.15
    ff_dim = 2048             # 扩展FFN维度
    
    # 训练参数
    lr = 0.003
    epochs = 50
    num_neg = 100             # 大批次负采样
    top_k = 10
    weight_decay = 1e-6
    warmup_steps = 10000
    
    # 高级优化
    label_smoothing = 0.1
    clip_grad = 5.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.item2idx = {}
        self.item_freq = None

    def load_data(self, train_path, test_path):
        # 并行化数据加载
        train = pd.read_csv(train_path, usecols=['parent_asin', 'history'])
        test = pd.read_csv(test_path, usecols=['parent_asin', 'history'])
        
        # 内存优化处理
        df = pd.concat([train, test])
        df['history'] = df['history'].apply(
            lambda x: [i for i in str(x).split() if len(i) == 10][-self.config.max_len*2:]
        )
        
        # 构建高效词典
        item_counter = Counter()
        for seq in df['history']:
            item_counter.update(seq)
        
        # 过滤低频项 (出现次数<5)
        item_counter = {k:v for k,v in item_counter.items() if v >= 5}
        self.item2idx = {item: i+1 for i, item in enumerate(item_counter)}
        self.num_items = len(self.item2idx) + 1
        
        # 构建序列数据集
        def process_partition(df):
            sequences = []
            for _, row in df.iterrows():
                seq = [self.item2idx.get(i, 0) for i in row['history']]
                target = self.item2idx.get(row['parent_asin'], 0)
                if target == 0: continue
                
                # 滑动窗口生成
                for i in range(2, len(seq)):
                    valid_seq = seq[max(0, i-self.config.max_len):i]
                    if len(valid_seq) < 3: continue
                    
                    pad_len = self.config.max_len - len(valid_seq)
                    sequences.append({
                        'seq': [0]*pad_len + valid_seq,
                        'target': target
                    })
            return sequences
        
        # 并行处理数据
        train_seqs = process_partition(train)
        test_seqs = process_partition(test)
        
        return train_seqs, test_seqs

class MegaSASRec(nn.Module):
    def __init__(self, config, num_items):
        super().__init__()
        self.config = config
        
        # 增强嵌入层
        self.item_emb = nn.Embedding(num_items, config.embed_dim, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.Tensor(config.max_len, config.embed_dim))
        
        # 深度Transformer结构
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                batch_first=True,
                activation='gelu',
                norm_first=True  # 前置LayerNorm
            ) for _ in range(config.num_blocks)
        ])
        
        # 正则化层
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # 预测头
        self.fc = nn.Linear(config.embed_dim, num_items)
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_normal_(self.item_emb.weight)
        nn.init.normal_(self.pos_emb, mean=0, std=0.02)
        nn.init.kaiming_normal_(self.fc.weight)
        
    def forward(self, seq):
        batch_size, seq_len = seq.size()
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0)
        
        # 混合精度计算
        with autocast(enabled=self.config.mixed_precision):
            # 嵌入融合
            item_emb = self.item_emb(seq)
            pos_emb = self.pos_emb[positions]
            seq_emb = self.norm(item_emb + pos_emb)
            
            # 注意力掩码
            src_key_padding_mask = (seq == 0)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, 
                                      dtype=torch.bool, device=seq.device), diagonal=1)
            
            # 深度Transformer处理
            for layer in self.encoder:
                seq_emb = layer(
                    src=seq_emb,
                    src_mask=causal_mask,
                    src_key_padding_mask=src_key_padding_mask
                )
            
            # 序列池化
            seq_emb = seq_emb[:, -1, :]
            return self.fc(self.dropout(seq_emb))

def evaluate(model, dataloader):
    model.eval()
    hits = 0
    ndcg = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            seq = batch['seq'].to(Config.device)
            targets = batch['target'].to(Config.device)
            
            logits = model(seq)
            scores = torch.softmax(logits, dim=1)
            scores[:, 0] = -1e9  # 过滤padding项
            
            _, topk = torch.topk(scores, k=Config.top_k, dim=1)
            
            # 并行计算指标
            hit_matrix = (topk == targets.unsqueeze(1))
            hits += hit_matrix.any(dim=1).sum().item()
            
            # 计算NDCG
            ranks = hit_matrix.nonzero()[:, 1] + 1
            ndcg += torch.sum(1 / torch.log2(ranks.float() + 1)).item()
    
    hit_rate = hits / len(dataloader.dataset)
    ndcg_score = ndcg / len(dataloader.dataset)
    return hit_rate, ndcg_score

def get_collate_fn(processor):
    def collate_fn(batch):
        seqs = [x['seq'] for x in batch]
        targets = [x['target'] for x in batch]
        
        # 高效负采样
        neg_samples = torch.randint(
            1, processor.num_items, 
            (len(batch)*Config.num_neg,),
            dtype=torch.long
        )
        
        # 合并正负样本
        all_seqs = []
        all_targets = []
        for i in range(len(batch)):
            all_seqs.append(seqs[i])
            all_targets.append(targets[i])
            for j in range(Config.num_neg):
                all_seqs.append(seqs[i])
                all_targets.append(neg_samples[i*Config.num_neg + j])
        
        return {
            'seq': torch.tensor(all_seqs, dtype=torch.long),
            'target': torch.tensor(all_targets, dtype=torch.long)
        }
    return collate_fn

def main():
    cfg = Config()
    writer = SummaryWriter()
    
    # 数据加载
    processor = DataProcessor(cfg)
    train_data, test_data = processor.load_data(
        "data/Video_Games_train.csv",
        "data/Video_Games_test.csv"
    )
    
    print(f"Total Items: {processor.num_items}")
    print(f"Train Samples: {len(train_data):,}")
    print(f"Test Samples: {len(test_data):,}")
    
    # 数据加载器优化
    collate_fn = get_collate_fn(processor)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=cfg.batch_size*2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 初始化模型
    model = MegaSASRec(cfg, processor.num_items).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.mixed_precision)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=cfg.lr,
        total_steps=cfg.epochs * len(train_loader),
        pct_start=0.1
    )
    
    # 带标签平滑的损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    
    # 训练循环
    best_ndcg = 0
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        
        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch in progress:
            seq = batch['seq'].to(cfg.device, non_blocking=True)
            targets = batch['target'].to(cfg.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 混合精度前向
            with autocast(enabled=cfg.mixed_precision):
                logits = model(seq)
                loss = criterion(logits, targets)
            
            # 梯度缩放和反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            progress.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
        
        # 评估
        hr, ndcg = evaluate(model, test_loader)
        writer.add_scalar('HR@10', hr, epoch)
        writer.add_scalar('NDCG@10', ndcg, epoch)
        print(f"Epoch {epoch+1}: HR@10={hr:.4f}, NDCG@10={ndcg:.4f}")
        
        # 保存最佳模型
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            torch.save(model.state_dict(), f"best_model_ndcg{ndcg:.4f}.pth")
            if ndcg > 0.8:
                break
    
    print(f"\nBest NDCG@10: {best_ndcg:.4f}")
    writer.close()

if __name__ == "__main__":
    main()
