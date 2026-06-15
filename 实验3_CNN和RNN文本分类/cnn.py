import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class Prep:
    def __init__(self, max_len=50):
        self.w2i = {"<PAD>": 0, "<UNK>": 1}
        self.i2w = {}
        self.max_len = max_len

    def build(self, sents):
        for s in sents:
            for w in s.split():
                if w not in self.w2i:
                    self.w2i[w] = len(self.w2i)
        self.i2w = {i: w for w, i in self.w2i.items()}

    def sent2id(self, s):
        ids = [self.w2i.get(w, 1) for w in s.split()]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        return ids[:self.max_len]

    def size(self):
        return len(self.w2i)


class TextDataset(Dataset):
    def __init__(self, x, y):
        self.x = [torch.tensor(i, dtype=torch.long) for i in x]
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class SelfAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden, 1)

    def forward(self, x):
        scores = self.attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class CNN(nn.Module):
    def __init__(self, vocab, embed, num_class, num_filters=128, filter_sizes=(2, 3, 4)):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed, padding_idx=0)
        self.emb_drop = nn.Dropout(0.2)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed, num_filters, fs, padding=fs // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Conv1d(num_filters, num_filters, fs, padding=fs // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU()
            ) for fs in filter_sizes
        ])

        total_filters = len(filter_sizes) * num_filters
        self.attention = SelfAttention(total_filters)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(total_filters, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_class)
        )

    def forward(self, x):
        x = self.emb(x)
        x = self.emb_drop(x).transpose(1, 2)

        conv_outs = []
        for conv in self.convs:
            c = conv(x)  # [batch, num_filters, seq_len']
            c = nn.functional.adaptive_max_pool1d(c, 1).squeeze(-1)  # [batch, num_filters]
            conv_outs.append(c)

        x = torch.cat(conv_outs, dim=1)  # [batch, total_filters]
        return self.fc(x)


class Metrics:
    @staticmethod
    def compute(preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)
        acc = (preds == labels).mean()

        classes = np.unique(labels)
        precision, recall, f1 = {}, {}, {}

        for c in classes:
            tp = ((preds == c) & (labels == c)).sum()
            fp = ((preds == c) & (labels != c)).sum()
            fn = ((preds != c) & (labels == c)).sum()

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0

            precision[c] = p
            recall[c] = r
            f1[c] = f

        macro_p = np.mean(list(precision.values()))
        macro_r = np.mean(list(recall.values()))
        macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0

        return {"acc": acc, "precision": precision, "recall": recall, "f1": f1,
                "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1}


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


class CNNTextClassifier:
    def __init__(self, vocab_size, embed_dim=128, num_class=2, num_filters=128,
                 filter_sizes=(2, 3, 4), device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN(vocab_size, embed_dim, num_class, num_filters, filter_sizes).to(self.device)
        self.prep = None
        self.class_weights = None

    def compute_class_weights(self, labels):
        counter = {}
        for l in labels:
            counter[l] = counter.get(l, 0) + 1
        total = len(labels)
        weights = []
        for i in range(len(counter)):
            w = total / (len(counter) * counter[i])
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32).to(self.device)

    def fit(self, train_sents, labels, epochs=20, lr=1e-3, batch_size=32,
            val_sents=None, val_labels=None, early_stop_patience=7):
        self.prep = Prep()
        self.prep.build(train_sents)

        x = [self.prep.sent2id(s) for s in train_sents]
        y = labels
        dataset = TextDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.class_weights = self.compute_class_weights(y)
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        early_stopping = EarlyStopping(patience=early_stop_patience)

        print(f"\n========== CNN 训练 (Device: {self.device}) ==========")
        print(f"词汇表: {self.prep.size()} | 类别数: {len(set(labels))}")

        best_val_f1 = 0
        for e in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                opt.zero_grad()
                out = self.model(x_batch)
                loss = loss_fn(out, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()

                total_loss += loss.item() * x_batch.size(0)
                pred = out.argmax(1)
                correct += (pred == y_batch).sum().item()
                total += y_batch.size(0)

            scheduler.step()
            acc = correct / total
            avg_loss = total_loss / total

            msg = f"Epoch {e + 1:2d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}"

            if val_sents and val_labels:
                val_metrics = self.evaluate(val_sents, val_labels)
                val_f1 = val_metrics['macro_f1']
                msg += f" | Val Acc: {val_metrics['acc']:.4f} | Val F1: {val_f1:.4f}"

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1

                val_loss = 1 - val_f1
                if early_stopping.step(val_loss):
                    print(f"  Early stopping at epoch {e + 1}")
                    break

            if (e + 1) % 5 == 0 or e == 0:
                print(msg)

        print(f"Best Val F1: {best_val_f1:.4f}")

    def predict(self, sents, batch_size=64):
        self.model.eval()
        results = []
        all_ids = [self.prep.sent2id(s) for s in sents]

        with torch.no_grad():
            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i:i + batch_size]
                x = torch.tensor(batch_ids, dtype=torch.long).to(self.device)
                out = self.model(x)
                results.extend(out.argmax(1).cpu().tolist())

        return results

    def predict_proba(self, sents, batch_size=64):
        self.model.eval()
        results = []
        all_ids = [self.prep.sent2id(s) for s in sents]

        with torch.no_grad():
            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i:i + batch_size]
                x = torch.tensor(batch_ids, dtype=torch.long).to(self.device)
                out = self.model(x)
                probs = torch.softmax(out, dim=1)
                results.extend(probs.cpu().tolist())

        return results

    def evaluate(self, test_sents, labels, batch_size=64):
        preds = self.predict(test_sents, batch_size)
        return Metrics.compute(preds, labels)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "prep": self.prep,
            "class_weights": self.class_weights
        }, path)
        print(f"模型已保存到 {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.prep = ckpt["prep"]
        if "class_weights" in ckpt:
            self.class_weights = ckpt["class_weights"]
        print(f"模型已加载")


def load_icwb2_sentences(data_dir="icwb2-data", max_lines=5000, min_len=3, max_len=30):
    all_sents = []
    sources = ["msr", "pku", "cityu"]

    for src in sources:
        filepath = f"{data_dir}/training/{src}_training.utf8"
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                count = 0
                for line in f:
                    if count >= max_lines // len(sources):
                        break
                    line = line.strip()
                    words = line.split()
                    if min_len <= len(words) <= max_len:
                        all_sents.append((line, src))
                        count += 1
        except FileNotFoundError:
            continue

    print(f"加载 ICWB2 数据: {len(all_sents)} 条 | 来源: {set(s[1] for s in all_sents)}")
    return all_sents


def create_classification_data(sents_with_source, num_classes=3, train_ratio=0.8):
    random.shuffle(sents_with_source)

    by_source = {}
    for sent, src in sents_with_source:
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(sent)

    train_sents, train_labels = [], []
    test_sents, test_labels = [], []

    class_names = sorted(list(by_source.keys()))[:num_classes]

    for idx, cls in enumerate(class_names):
        data = by_source[cls]
        split_idx = int(len(data) * train_ratio)
        train_sents.extend(data[:split_idx])
        train_labels.extend([idx] * split_idx)
        test_sents.extend(data[split_idx:])
        test_labels.extend([idx] * (len(data) - split_idx))

    combined = list(zip(train_sents, train_labels))
    random.shuffle(combined)
    train_sents, train_labels = zip(*combined)

    combined = list(zip(test_sents, test_labels))
    random.shuffle(combined)
    test_sents, test_labels = zip(*combined)

    return list(train_sents), list(train_labels), list(test_sents), list(test_labels), class_names


if __name__ == "__main__":
    print("========== 加载 ICWB2 数据 ==========")
    raw_sents = load_icwb2_sentences(data_dir="icwb2-data", max_lines=6000, min_len=5, max_len=25)

    if len(raw_sents) == 0:
        print("ICWB2 数据未找到，使用示例数据")
        sents = [
            "今天 天气 很好", "明天 天气 晴朗", "天气 下雨 了", "今天 很 热",
            "外面 刮风 了", "天气 很 舒服", "天气 不错", "下雨 了",
            "我 在 学习", "机器 学习 很 有趣", "深度 学习 好 难",
            "我 爱 编程", "算法 很 重要", "学习 使 我 快乐",
            "学习 深度 学习", "我爱自然语言处理"
        ]
        labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

        clf = CNNTextClassifier(vocab_size=100, embed_dim=64, num_class=2, num_filters=128)
        clf.fit(sents, labels, epochs=20, lr=0.001, batch_size=4)

        print("\n========== 测试预测 ==========")
        test_sents = ["今天 天气 不错", "我 在 努力 学习"]
        preds = clf.predict(test_sents)
        for s, p in zip(test_sents, preds):
            label = "天气" if p == 0 else "学习"
            print(f"输入：{s} → 分类：{label}")
    else:
        print("\n========== 构建分类任务 ==========")
        print("类别说明: msr=微软亚洲研究院, pku=北京大学, cityu=香港城市大学")
        train_sents, train_labels, test_sents, test_labels, class_names = create_classification_data(raw_sents, num_classes=3)

        print(f"训练集: {len(train_sents)} 条")
        print(f"测试集: {len(test_sents)} 条")
        print(f"类别: {class_names}")

        clf = CNNTextClassifier(vocab_size=20000, embed_dim=64, num_class=3, num_filters=128)
        clf.fit(train_sents, train_labels, epochs=30, lr=0.001, batch_size=32,
                val_sents=test_sents, val_labels=test_labels, early_stop_patience=7)

        print("\n========== 测试预测 ==========")
        test_samples = test_sents[:6]
        true_labels = test_labels[:6]
        preds = clf.predict(test_samples)

        for s, p, t in zip(test_samples, preds, true_labels):
            print(f"输入：{s[:40]}...")
            print(f"  预测: {class_names[p]} | 标准: {class_names[t]} | {'OK' if p == t else 'X'}")

        print("\n========== 评估指标 ==========")
        metrics = clf.evaluate(test_sents, test_labels)
        print(f"Accuracy: {metrics['acc']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")

        for c in metrics['f1']:
            print(f"  Class {class_names[c]}: P={metrics['precision'][c]:.4f} R={metrics['recall'][c]:.4f} F1={metrics['f1'][c]:.4f}")

        clf.save("cnn_model.pt")
