import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import random

# 固定随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed(42) if torch.cuda.is_available() else None

class Prep:
    def __init__(self, min_freq=1):
        self.w2i = {"<PAD>": 0, "<UNK>": 1}
        self.i2w = {}
        self.word_counts = Counter()
        self.min_freq = min_freq  # 过滤低频词

    def build(self, sents):
        # 统计词频
        for s in sents:
            self.word_counts.update(s.split())
        # 过滤低频词 + 构建词表
        for word, cnt in self.word_counts.items():
            if cnt >= self.min_freq and word not in self.w2i:
                self.w2i[word] = len(self.w2i)
        self.i2w = {i: w for w, i in self.w2i.items()}

    def sent2id(self, sent):
        return [self.w2i.get(w, 1) for w in sent.split()]

    def size(self):
        return len(self.w2i)

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.emb_center = nn.Embedding(vocab_size, embed_dim)
        self.emb_context = nn.Embedding(vocab_size, embed_dim)
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, center, context, negative):
        center_emb = self.emb_center(center)      # [batch, dim]
        context_emb = self.emb_context(context)   # [batch, dim]
        negative_emb = self.emb_context(negative) # [batch, k, dim]

        # 正样本得分
        pos_score = torch.sum(center_emb * context_emb, dim=1)
        # 负样本得分
        neg_score = torch.bmm(negative_emb, center_emb.unsqueeze(-1)).squeeze(-1)
        
        loss = -self.log_sigmoid(pos_score).mean() - self.log_sigmoid(-neg_score).mean()
        return loss

    def emb(self):
        return self.emb_center.weight.data.cpu().numpy()

# 批处理数据生成
def gen_data(sents, prep, window_size=2):
    data = []
    for sent in sents:
        ids = prep.sent2id(sent)
        for i, center in enumerate(ids):
            start = max(0, i - window_size)
            end = min(len(ids), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    data.append((center, ids[j]))
    return data

def init_neg_prob(prep):
    vocab_size = prep.size()
    word_freq = np.array([prep.word_counts.get(prep.i2w[i], 1) for i in range(vocab_size)])
    word_freq = word_freq ** 0.75
    return word_freq / word_freq.sum()

# 批处理负采样
def neg_samples_batch(prob, vocab_size, targets, k=5):
    neg_list = []
    for t in targets:
        negs = []
        while len(negs) < k:
            s = np.random.choice(vocab_size, p=prob)
            if s != t:
                negs.append(s)
        neg_list.append(negs)
    return neg_list

# 批处理训练（核心优化）
def train(model, data, prep, epochs=10, lr=0.01, n_neg=5, batch_size=32):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    device = torch.device("cpu")  # 强制使用 CPU
    model.to(device)
    
    vocab_size = prep.size()
    prob = init_neg_prob(prep)
    data = np.array(data)
    
    print(f"\n开始训练... 设备: {device} | 批次大小: {batch_size}")
    for epoch in range(epochs):
        total_loss = 0
        np.random.shuffle(data)
        
        # 批处理迭代
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            centers = batch[:, 0]
            contexts = batch[:, 1]
            negatives = neg_samples_batch(prob, vocab_size, centers, n_neg)
            
            # 转张量
            centers = torch.tensor(centers, dtype=torch.long).to(device)
            contexts = torch.tensor(contexts, dtype=torch.long).to(device)
            negatives = torch.tensor(negatives, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            loss = model(centers, contexts, negatives)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch)
        
        avg_loss = total_loss / len(data)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

# 后续工具函数完全保留原功能
def sent_vec(model, ids):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        vec = model.emb_center(torch.tensor(ids, dtype=torch.long).to(device))
        return vec.mean(0).cpu().numpy().reshape(1, -1) if len(vec) else np.zeros(model.emb_center.embedding_dim)

def sim(v1, v2):
    return cosine_similarity(v1, v2)[0][0]

def save_emb(model, prep, filename):
    embeddings = model.emb()
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{len(prep.w2i)} {model.emb_center.embedding_dim}\n")
        for word, idx in prep.w2i.items():
            f.write(f"{word} {' '.join(map(str, embeddings[idx]))}\n")
    print(f"\n词向量已保存到 {filename}")

def word_vec(model, prep, word):
    idx = prep.w2i.get(word, 1)
    return model.emb_center.weight.data[idx].cpu().numpy().reshape(1, -1)

def eval_sim(model, prep):
    print("\n========== 词相似度评估 ==========")
    test_words = [("我","是"),("是","的"),("的","在"),("在","和"),("中国","国家")]
    for w1, w2 in test_words:
        if w1 in prep.w2i and w2 in prep.w2i:
            print(f"{w1} 与 {w2} 相似度: {sim(word_vec(model,prep,w1), word_vec(model,prep,w2)):.4f}")

def find_similar(model, prep, word, top_k=5):
    target_vec = word_vec(model, prep, word)
    similarities = []
    for i in range(prep.size()):
        w = prep.i2w[i]
        if w not in [word, "<PAD>", "<UNK>"]:
            vec = model.emb_center.weight.data[i].cpu().numpy().reshape(1,-1)
            similarities.append((w, sim(target_vec, vec)))
    return sorted(similarities, key=lambda x:-x[1])[:top_k]

def load_sents(filepath, max_lines=1000):
    sents = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines: break
                line = line.strip()
                if line: sents.append(line)
        print(f"加载数据 {len(sents)} 条")
    except:
        print("使用内置示例数据")
        sents = [
            "我 爱 北京 天安门", "我 爱 中国", "北京 是 中国 的 首都",
            "今天 天气 很好", "明天 天气 晴朗", "自然 语言 处理 是 人工智能 的 分支"
        ]
    return sents

if __name__ == "__main__":
    # 1. 加载数据
    sents = load_sents("icwb2-data/training/msr_training.utf8", max_lines=500)
    
    # 2. 预处理
    prep = Prep(min_freq=2)
    prep.build(sents)
    print(f"词表大小: {prep.size()}")
    
    # 3. 生成训练数据
    data = gen_data(sents, prep, window_size=2)
    print(f"训练样本数: {len(data)}")
    
    # 4. 训练
    model = SkipGram(prep.size(), embed_dim=32)
    train(model, data, prep, epochs=10, lr=0.05, n_neg=5, batch_size=32)
    
    # 5. 评估与测试
    embeddings = model.emb()
    print("\n========== 词向量示例 ==========")
    for w in ["我","是","的","在","和"]:
        if w in prep.w2i:
            print(f"{w}: {embeddings[prep.w2i[w]][:5]}")
    
    # 句子相似度
    print("\n========== 句子相似度 ==========")
    test_pairs = [("我 是 一个 学生","他 是 一个 老师"),("中国 是 一个 国家","北京 是 中国 的 首都")]
    for s1,s2 in test_pairs:
        v1 = sent_vec(model, prep.sent2id(s1))
        v2 = sent_vec(model, prep.sent2id(s2))
        print(f"{s1} vs {s2}: {sim(v1, v2):.4f}")
    
    eval_sim(model, prep)
    
    # 相似词
    print("\n========== 相似词 ==========")
    for w in ["我","是","的","在"]:
        if w in prep.w2i:
            similar = find_similar(model, prep, w, 3)
            print(f"{w}: {similar}")
    
    save_emb(model, prep, "word_vectors1.txt")