import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Config:
    max_len = 50
    embed_dim = 128
    num_heads = 8
    num_blocks = 3
    dropout = 0.2
    batch_size = 256
    lr = 0.001
    epochs = 50
    num_neg = 20
    weight_decay = 1e-5
    top_k = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.item2idx = {}
        self.item_freq = None

    def load_data(self, train_path, test_path):
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        df = pd.concat([train, test])
        
        df['history'] = df['history'].apply(
            lambda x: [i for i in str(x).split() if len(i) == 10 and i.startswith('B')]
        )
        df = df[df['history'].map(len) > 3]
        
        item_counter = Counter()
        for seq in df['history']:
            item_counter.update(seq)
        self.item2idx = {item: i+1 for i, (item, _) in enumerate(item_counter.most_common())}
        self.num_items = len(self.item2idx) + 1
        
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        
        train_seqs = self._create_sequences(train)
        test_seqs = self._create_sequences(test, is_train=False)
        
        self._calc_sampled_prob(item_counter)
        return train_seqs, test_seqs

    def _create_sequences(self, df, is_train=True):
        sequences = []
        for _, row in df.iterrows():
            seq = [self.item2idx[i] for i in row['history'] if i in self.item2idx]
            target = self.item2idx.get(row['parent_asin'], 0)
            if target == 0 or len(seq) < 4:
                continue
            
            for i in range(3, len(seq)):
                if is_train and random.random() < 0.3:
                    continue
                input_seq = seq[max(0, i-self.config.max_len):i]
                pad_len = self.config.max_len - len(input_seq)
                sequences.append({
                    'seq': [0]*pad_len + input_seq[-self.config.max_len:],
                    'target': target
                })
        return sequences

    def _calc_sampled_prob(self, counter):
        temperature = 0.75
        freqs = np.array([counter.get(item, 0) for item in self.item2idx.keys()])
        freqs = np.power(freqs, temperature)
        self.sample_probs = freqs / freqs.sum()

class SASRec(nn.Module):
    def __init__(self, config, num_items):
        super().__init__()
        self.config = config
        
        self.item_emb = nn.Embedding(num_items, config.embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(config.max_len, config.embed_dim)
        
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)
        
        # 修正Transformer层参数
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.num_heads,
                dim_feedforward=config.embed_dim*4,
                dropout=config.dropout,
                batch_first=True,
                activation='gelu'
            ) for _ in range(config.num_blocks)
        ])
        
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.embed_dim, num_items)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, seq):
        batch_size, seq_len = seq.size()
        positions = torch.arange(seq_len, device=seq.device).expand(batch_size, seq_len)
        
        item_emb = self.item_emb(seq)
        pos_emb = self.pos_emb(positions)
        seq_emb = self.layer_norm(item_emb + pos_emb)
        
        # 修正mask参数传递
        src_key_padding_mask = (seq == 0)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, 
                                  dtype=torch.bool, device=seq.device), diagonal=1)
        
        for layer in self.encoder:
            seq_emb = layer(
                src=seq_emb,
                src_mask=causal_mask,  # 修正参数名
                src_key_padding_mask=src_key_padding_mask
            )
        
        seq_emb = seq_emb[:, -1, :]
        return self.fc(self.dropout(seq_emb))

def evaluate(model, dataloader, processor):
    model.eval()
    hits, ndcg = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            seq = batch['seq'].to(Config.device)
            targets = batch['target'].to(Config.device)
            
            logits = model(seq)
            scores = torch.softmax(logits, dim=1)
            scores[:, 0] = -1e9  # 过滤padding项
            
            _, top_items = torch.topk(scores, k=Config.top_k, dim=1)
            
            for target, items in zip(targets, top_items):
                if target in items:
                    hits += 1
                    rank = (items == target).nonzero().item() + 1
                    ndcg += 1 / np.log2(rank + 1)
    
    hit_rate = hits / len(dataloader.dataset)
    ndcg_score = ndcg / len(dataloader.dataset)
    return hit_rate, ndcg_score

def get_collate_fn(processor):
    def collate_fn(batch):
        seqs = [x['seq'] for x in batch]
        targets = [x['target'] for x in batch]
        
        neg_samples = np.random.choice(
            list(processor.item2idx.values()),
            size=len(batch)*Config.num_neg,
            p=processor.sample_probs
        )
        
        all_seqs = []
        all_targets = []
        for i in range(len(batch)):
            all_seqs.append(seqs[i])
            all_targets.append(targets[i])
            for j in range(Config.num_neg):
                all_seqs.append(seqs[i])
                all_targets.append(neg_samples[i*Config.num_neg + j])
        
        seq_tensor = torch.tensor(all_seqs, dtype=torch.long)
        target_tensor = torch.tensor(all_targets, dtype=torch.long)
        return {'seq': seq_tensor, 'target': target_tensor}
    return collate_fn

def main():
    cfg = Config()
    writer = SummaryWriter()
    
    processor = DataProcessor(cfg)
    train_data, test_data = processor.load_data(
        "data/Video_Games_train.csv",
        "data/Video_Games_test.csv"
    )
    
    print(f"Items: {processor.num_items}")
    print(f"Train sequences: {len(train_data)}")
    print(f"Test sequences: {len(test_data)}")
    
    collate_fn = get_collate_fn(processor)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=cfg.batch_size*2,
        collate_fn=collate_fn
    )
    
    model = SASRec(cfg, processor.num_items).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_ndcg = 0
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch in progress:
            seq = batch['seq'].to(cfg.device, non_blocking=True)
            targets = batch['target'].to(cfg.device, non_blocking=True)
            
            optimizer.zero_grad()
            logits = model(seq)
            loss = criterion(logits, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        
        hr, ndcg = evaluate(model, test_loader, processor)
        writer.add_scalar('Test/HR@10', hr, epoch)
        writer.add_scalar('Test/NDCG@10', ndcg, epoch)
        print(f"Epoch {epoch+1}: HR@10={hr:.4f}, NDCG@10={ndcg:.4f}")
        
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            torch.save(model.state_dict(), f'best_model_ndcg{ndcg:.4f}.pth')
            if ndcg > 0.8:
                break

    print(f"\nBest NDCG@10: {best_ndcg:.4f}")
    writer.close()

if __name__ == "__main__":
    main()