import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

class Prep:
    def __init__(self):
        self.c2i = {"<PAD>":0, "<UNK>":1}
        self.t2i = {"B":0, "I":1, "E":2, "S":3}
        self.i2t = {v:k for k,v in self.t2i.items()}
        self.unk_rate = 0.0

    def gen_tags(self, sent):
        tags = []
        words = sent.strip().split()
        for w in words:
            if len(w) == 1:
                tags.append("S")
            else:
                tags.append("B")
                tags.extend(["I"]*(len(w)-2))
                tags.append("E")
        return tags, "".join(words)

    def build_vocab(self, texts):
        for t in texts:
            for c in t:
                if c not in self.c2i:
                    self.c2i[c] = len(self.c2i)

    def text2id(self, text):
        return [self.c2i.get(c,1) for c in text]

    def tag2id(self, tags):
        return [self.t2i[t] for t in tags]

    def calculate_unk_rate(self, texts):
        total_chars = 0
        unk_chars = 0
        for t in texts:
            for c in t:
                total_chars += 1
                if c not in self.c2i:
                    unk_chars += 1
        self.unk_rate = unk_chars / total_chars if total_chars > 0 else 0
        return self.unk_rate

class DataSet(Dataset):
    def __init__(self, x, y):
        self.x = [torch.tensor(i) for i in x]
        self.y = [torch.tensor(i) for i in y]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class Model(nn.Module):
    def __init__(self, vocab, embed, hidden, num_tag):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed, 0)
        self.fc1 = nn.Linear(embed, hidden)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden, num_tag)

    def forward(self, x):
        x = self.emb(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        return self.fc2(x)

def train(model, loader, loss_fn, opt, epochs=30):
    model.train()
    print("\n开始训练...")
    for e in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for x, y in loader:
            opt.zero_grad()
            out = model(x)
            out = out.view(-1, out.shape[-1])
            y = y.view(-1)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()*x.size(0)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
        acc = correct/total
        avg_loss=total_loss/total
        print(f"Epoch {e+1} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

def pred(model, text, prep):
    model.eval()
    with torch.no_grad():
        ids = prep.text2id(text)
        x = torch.tensor([ids])
        out = model(x)
        pred_ids = out.argmax(-1)[0].tolist()
        tags = [prep.i2t[i] for i in pred_ids]

    res = []
    word = ""
    for c, t in zip(text, tags):
        word += c
        if t in ("S", "E"):
            res.append(word)
            word = ""
    if word:
        res.append(word)
    return res

def load_training_data(filepath, max_lines=2000):
    sents = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines: break
                line = line.strip()
                if line: sents.append(line)
        print(f"加载训练数据 {len(sents)} 条")
    except:
        print("加载失败，使用示例数据")
        sents = ["我 爱 北京 天安门", "自然 语言 处理 很 有趣", "机器 学习"]
    return sents

def load_test_data(test_path, gold_path):
    test_sents, gold_sents = [], []
    try:
        with open(test_path, 'r', encoding='utf-8') as f:
            test_sents = [line.strip() for line in f if line.strip()]
        with open(gold_path, 'r', encoding='utf-8') as f:
            gold_sents = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"加载测试数据失败: {e}")
    return test_sents, gold_sents

def evaluate_model(model, test_sents, gold_sents, prep, max_samples=200):
    total_f1 = 0
    cnt = 0
    p_total = 0
    r_total = 0

    print("\n========== 模型评估 ==========")

    for idx, (t, g) in enumerate(zip(test_sents, gold_sents)):
        if idx >= max_samples:
            break

        text = t.replace(" ", "")
        gold = g.split()

        pred_words = pred(model, text, prep)

        a = set(pred_words)
        b = set(gold)
        cr = len(a & b)

        p = cr / len(a) if a else 0
        r = cr / len(b) if b else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        p_total += p
        r_total += r
        total_f1 += f1
        cnt += 1

        if idx < 10:
            print(f"\n[{idx+1}] 输入: {text[:30]}...")
            print(f"    标准: {' / '.join(gold[:10])}{'...' if len(gold)>10 else ''}")
            print(f"    预测: {' / '.join(pred_words[:10])}{'...' if len(pred_words)>10 else ''}")
            print(f"    P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f}")

    if cnt > 0:
        avg_p = p_total / cnt
        avg_r = r_total / cnt
        avg_f1 = total_f1 / cnt
        print(f"\n========== 平均性能 (共{cnt}条) ==========")
        print(f"Precision: {avg_p:.4f}")
        print(f"Recall:    {avg_r:.4f}")
        print(f"F1 Score:  {avg_f1:.4f}")
    else:
        print("没有有效的评估样本")

if __name__ == "__main__":
    train_path = "icwb2-data/training/msr_training.utf8"
    test_path = "icwb2-data/testing/msr_test.utf8"
    gold_path = "icwb2-data/gold/msr_test_gold.utf8"

    print("========== 加载数据 ==========")
    train_sents = load_training_data(train_path, max_lines=2000)
    test_sents, gold_sents = load_test_data(test_path, gold_path)

    if len(train_sents) == 0:
        print("训练数据为空，退出")
        exit(1)

    print(f"\n测试数据: {len(test_sents)} 条")
    print(f"标准分词: {len(gold_sents)} 条")

    prep = Prep()
    all_text, all_tags = [], []
    valid_count = 0

    for s in train_sents:
        tg, tx = prep.gen_tags(s)
        if len(tg) == len(tx):
            all_text.append(tx)
            all_tags.append(tg)
            valid_count += 1

    print(f"有效训练样本: {valid_count}/{len(train_sents)}")

    prep.build_vocab(all_text)
    unk_rate = prep.calculate_unk_rate(all_text)
    print(f"词汇表大小: {len(prep.c2i)}")
    print(f"UNK率: {unk_rate:.2%}")

    x = [prep.text2id(t) for t in all_text]
    y = [prep.tag2id(t) for t in all_tags]

    dataset = DataSet(x, y)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = Model(len(prep.c2i), 64, 256, 4)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    train(model, loader, loss_fn, opt, epochs=20)

    if len(test_sents) > 0 and len(gold_sents) > 0:
        evaluate_model(model, test_sents, gold_sents, prep, max_samples=200)

    print("\n========== 分词测试 ==========")
    for s in ["我爱北京天安门", "今天天气很好", "我在学习深度学习"]:
        result = pred(model, s, prep)
        print(f"{s} → {' / '.join(result)}")
