"""本地问答引擎 - 数学计算 + 法律问答（两阶段检索）"""

import re
import math
import random
import json
import os
import pickle
import datetime

import jieba

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
LAW_FILE = os.path.join(DATA_DIR, "law_merged.json")
GRAPH_FILE = os.path.join(DATA_DIR, "law_graph.pkl")

# 全局变量
_questions = None  # 问题列表
_answers = None    # 答案列表
_graph = None      # 知识图谱
_law_idx = None    # 法律名称 -> [法条索引]

# ---- 同义词映射 ----
_SYN = {
    '什么是': ['什么是', '什么叫', '啥是', '是啥', '什么意思', '解释一下', '介绍一下'],
    '为什么': ['为什么', '为何', '为啥', '原因'],
    '怎么': ['怎么', '怎样', '如何', '咋', '咋样'],
    '多少': ['多少', '几', '几多'],
    '能': ['能', '可以', '能够', '可'],
    '要': ['要', '需要', '得', '必须'],
    '有': ['有', '存在', '具备', '拥有'],
    '没有': ['没有', '无', '不存在'],
    '知道': ['知道', '了解', '明白', '清楚', '晓得'],
    '处罚': ['处罚', '惩罚', '判刑', '量刑', '怎么判', '怎么罚'],
    '离婚': ['离婚', '解除婚姻', '分手', '婚姻破裂'],
    '合同': ['合同', '协议', '契约', '约定'],
    '赔偿': ['赔偿', '赔钱', '补偿', '索赔'],
    '工资': ['工资', '薪酬', '薪水', '报酬', '收入'],
    '辞退': ['辞退', '解雇', '开除', '裁员', '炒鱿鱼'],
    '酒驾': ['酒驾', '饮酒驾驶', '酒后驾驶', '喝酒开车'],
    '醉驾': ['醉驾', '醉酒驾驶', '危险驾驶'],
    '诈骗': ['诈骗', '骗钱', '被骗', '骗局', '欺诈'],
    '盗窃': ['盗窃', '偷窃', '偷东西', '小偷'],
    '抚养': ['抚养', '抚养权', '抚养费', '孩子归谁'],
    '继承': ['继承', '遗产', '遗嘱', '财产继承'],
    '侵权': ['侵权', '侵犯', '侵害'],
    '买房': ['买房', '房屋买卖', '购房', '商品房'],
    '租房': ['租房', '租赁', '房屋租赁', '出租'],
    '工伤': ['工伤', '工作中受伤', '职业伤害'],
    '加班': ['加班', '加班费', '加班工资'],
    '试用期': ['试用期', '试用', '实习期'],
    '消费者': ['消费者', '消费', '买家'],
    '退货': ['退货', '退款', '退换', '退钱'],
    '假货': ['假货', '假冒', '仿冒', '山寨'],
    '隐私': ['隐私', '个人信息', '信息泄露', '数据泄露'],
    '打架': ['打架', '斗殴', '动手', '打人'],
    '故意伤害': ['故意伤害', '蓄意伤人', '故意伤人', '蓄意伤害', '蓄意'],
    '借款': ['借款', '借钱', '欠钱', '欠款', '贷款', '民间借贷'],
    '定金': ['定金', '订金', '押金', '诚意金'],
    '物业': ['物业', '物业管理', '物业费'],
    '缓刑': ['缓刑', '判缓', '监外执行'],
    '自首': ['自首', '投案', '自新'],
    '帮信': ['帮信', '帮信罪', '帮助信息网络犯罪'],
    '寻衅滋事': ['寻衅滋事', '闹事', '挑衅', '惹事'],
    '取保候审': ['取保候审', '取保', '保释'],
    '防卫': ['防卫', '正当防卫', '自卫', '还手'],
    '防卫过当': ['防卫过当', '防卫过度', '反击过重'],
    '诉讼时效': ['诉讼时效', '过期', '超过时效', '时效'],
    '行为能力': ['行为能力', '民事行为能力', '行为能力人'],
    '著作权': ['著作权', '版权', '版权保护', '作品'],
    '知识产权': ['知识产权', '专利', '商标'],
    '结婚': ['结婚', '婚姻', '领证', '婚龄'],
    '财产分割': ['财产分割', '分财产', '分家产', '财产分配'],
    '交通事故': ['交通事故', '车祸', '撞车', '肇事'],
}

_STOP = set([
    '的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
    '或', '一个', '我们', '你们', '他们', '这个', '那个',
    '因为', '所以', '但是', '然而', '如果',
    '可以', '能够', '应该', '必须', '需要', '可能', '会',
    '不', '很', '也', '还', '又', '再', '更', '最', '太',
    '吧', '呢', '啊', '吗', '呀', '哦', '嗯',
    '请', '麻烦', '大家', '各位', '一下', '一下儿',
])

_GENERIC = {
    '怎么': 0.2, '怎么办': 0.2, '如何': 0.2, '怎样': 0.2,
    '什么是': 0.3, '什么叫': 0.3, '什么意思': 0.3,
    '为什么': 0.3, '为啥': 0.3, '为何': 0.3,
    '处罚': 0.2, '惩罚': 0.2, '判刑': 0.2,
    '多少': 0.3, '几': 0.3,
    '能': 0.3, '可以': 0.3, '要': 0.3, '有': 0.3,
    '知道': 0.3, '了解': 0.3, '没有': 0.3,
    '规定': 0.2, '法律': 0.2, '条例': 0.2,
    '注意': 0.1, '情形': 0.1, '条件': 0.2,
    '标准': 0.2, '范围': 0.2, '期限': 0.2,
    '权利': 0.2, '义务': 0.2, '责任': 0.2,
    '行为': 0.2, '情况': 0.1,
}


def _extract_kw(text: str) -> list[str]:
    text = text.strip()
    for target in sorted(_SYN, key=len, reverse=True):
        for syn in sorted(_SYN[target], key=len, reverse=True):
            if syn in text:
                text = text.replace(syn, target)
    words = list(jieba.cut(text))
    words = [w for w in words if w.strip() and w not in _STOP]
    words = [re.sub(r'[，,。.？?！!、\\/：:；;""''（）()【】《》]', '', w) for w in words]
    return [w for w in words if w]


def build_index():
    global _questions, _answers, _graph, _law_idx

    if not os.path.exists(LAW_FILE):
        print("[索引] 法律知识库不存在，跳过")
        return

    if os.path.exists(GRAPH_FILE):
        print("[索引] 加载法律图谱缓存...")
        with open(GRAPH_FILE, "rb") as f:
            data = pickle.load(f)
        _questions = data["questions"]
        _answers = data["answers"]
        _graph = data["graph"]
        _law_idx = data.get("law_idx", {})
        print(f"[索引] 加载完成: {len(_answers)} 条法律知识")
        return

    print("[索引] 构建法律知识图谱...")
    with open(LAW_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    _questions = [item["q"] for item in dataset]
    _answers = [item["a"] for item in dataset]
    _graph = []
    _law_idx = {}  # 法律名称 -> [法条索引列表]

    for i, item in enumerate(dataset):
        kw_weights = {}
        q_kw = _extract_kw(item["q"])
        for kw in q_kw:
            kw_weights[kw] = kw_weights.get(kw, 0) + 3.0
        for kw in q_kw:
            for target, syns in _SYN.items():
                if kw == target:
                    for syn in syns:
                        if syn != target:
                            kw_weights[syn] = kw_weights.get(syn, 0) + 2.5
        a_kw = _extract_kw(item["a"])[:10]
        for kw in a_kw:
            kw_weights[kw] = kw_weights.get(kw, 0) + 0.3
        _graph.append(kw_weights)

        # 构建法律名称索引
        if _is_article(i):
            m = re.match(r'根据《(.+?)》第', item["a"])
            if m:
                law_name = m.group(1)
                if law_name not in _law_idx:
                    _law_idx[law_name] = []
                _law_idx[law_name].append(i)

        if (i + 1) % 5000 == 0:
            print(f"  已处理 {i+1}/{len(dataset)}")

    with open(GRAPH_FILE, "wb") as f:
        pickle.dump({"questions": _questions, "answers": _answers, "graph": _graph, "law_idx": _law_idx}, f)

    print(f"[索引] 构建完成: {len(_answers)} 条法律知识, {len(_law_idx)} 部法律, 关键词总数: {sum(len(g) for g in _graph)}")


def _graph_search(question: str, top_k: int = 5) -> list[tuple[int, float]]:
    """图谱搜索，返回(索引, 得分)列表"""
    if _graph is None:
        return []

    q_kws = _extract_kw(question)
    if not q_kws:
        return []

    q_kw_weights = {}
    for kw in q_kws:
        q_kw_weights[kw] = _GENERIC.get(kw, 1.0)
    for kw in list(q_kw_weights.keys()):
        if kw in _GENERIC:
            continue
        for target, syns in _SYN.items():
            if kw == target:
                for syn in syns:
                    if syn not in q_kw_weights and syn not in _GENERIC:
                        q_kw_weights[syn] = 0.8

    scores = []
    query_len = len(q_kws)
    min_coverage = 0.5 if query_len <= 2 else 0.3

    for i, kw_map in enumerate(_graph):
        score = 0.0
        matched = 0
        for kw, q_weight in q_kw_weights.items():
            if kw in kw_map:
                score += kw_map[kw] * q_weight
                matched += 1
        if matched > 0:
            coverage = matched / query_len
            if coverage < min_coverage:
                continue
            density = matched / max(len(kw_map), 1)
            base_score = score * (0.6 * coverage + 0.4 * min(density * 5, 1.0))
            scores.append((i, base_score))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


def _is_article(idx: int) -> bool:
    """判断是否为法条（答案以"根据《"开头）"""
    return _answers[idx].startswith("根据《")


def _law_rag(question: str) -> str | None:
    """两阶段检索：咨询回答 + 法条引用组合"""
    results = _graph_search(question, top_k=10)
    if not results:
        return None

    # 分离法条和咨询回答
    articles = []
    advises = []
    for idx, score in results:
        if _is_article(idx):
            articles.append((idx, score))
        else:
            advises.append((idx, score))

    # 阶段1：找最佳咨询回答
    best_advise = None
    best_advise_score = 0
    if advises:
        best_advise_idx, best_advise_score = advises[0]
        best_advise = _answers[best_advise_idx]

    # 阶段2：从咨询回答中提取法律名称，反向搜索法条
    related_articles = []
    
    # 2a. 先提取法律名称
    law_names = set()
    if best_advise:
        for m in re.finditer(r'《(.+?)》', best_advise):
            law_names.add(m.group(1))
    for m in re.finditer(r'《(.+?)》', question):
        law_names.add(m.group(1))

    # 2b. 用直接匹配到的法条（需要和法律名称相关）
    for idx, score in articles[:3]:
        if score > 0.3:
            if law_names:
                if any(f'《{name}》' in _answers[idx] for name in law_names):
                    related_articles.append(_answers[idx])
            else:
                related_articles.append(_answers[idx])
    
    # 用法律名称+法条索引精确搜索法条
    if law_names and len(related_articles) < 3 and _law_idx:
        # 从咨询回答中提取法条引用（如"第九十一条"）
        art_refs = set()
        if best_advise:
            for m in re.finditer(r'第([一二三四五六七八九十百千零\d]+)条', best_advise):
                # 过滤掉占位符如"第X条"
                ref = m.group(1)
                if ref not in ('X', 'x') and not any(c in ref for c in '被处将面'):
                    art_refs.add(ref)
        
        for name in law_names:
            if len(related_articles) >= 3:
                break
            # 在该法律的法条中搜索
            indices = _law_idx.get(name, [])
            if not indices:
                continue
            # 优先匹配有具体条号的法条
            if art_refs:
                for ref_num in art_refs:
                    for idx in indices:
                        if f'第{ref_num}条' in _answers[idx] and _answers[idx] not in related_articles:
                            related_articles.append(_answers[idx])
                            if len(related_articles) >= 3:
                                break
                    if len(related_articles) >= 3:
                        break
            # 如果没有具体条号，用关键词在该法律的法条中搜索
            if not art_refs or len(related_articles) == 0:
                # 扩展查询关键词（同义词替换 + 分词展开）
                q_kws = set(_extract_kw(question))
                expanded = set(q_kws)
                for kw in q_kws:
                    for target, syns in _SYN.items():
                        if kw == target:
                            for syn in syns:
                                expanded.add(syn)
                                for w in jieba.cut(syn):
                                    if w.strip() and w not in _STOP:
                                        expanded.add(w)
                # 去除通用词
                expanded -= set(_GENERIC.keys())
                scored = []
                for idx in indices:
                    # 直接在法条文本中搜索关键词
                    text = _answers[idx]
                    hits = sum(1 for kw in expanded if kw in text)
                    if hits > 0:
                        scored.append((idx, hits))
                scored.sort(key=lambda x: -x[1])
                for idx, _ in scored[:3]:
                    if _answers[idx] not in related_articles:
                        related_articles.append(_answers[idx])
                        if len(related_articles) >= 3:
                            break

    # 后备：如果咨询回答中没有法律名称，用问题关键词在法条中搜索
    if not law_names and len(related_articles) == 0 and _law_idx:
        q_kws = set(_extract_kw(question))
        expanded = set(q_kws)
        for kw in q_kws:
            for target, syns in _SYN.items():
                if kw == target:
                    for syn in syns:
                        expanded.add(syn)
                        for w in jieba.cut(syn):
                            if w.strip() and w not in _STOP:
                                expanded.add(w)
        expanded -= set(_GENERIC.keys())
        # 只搜索包含核心关键词的法条（用图谱搜索缩小范围）
        art_results = _graph_search(question, top_k=20)
        scored = []
        for idx, _ in art_results:
            if not _is_article(idx):
                continue
            text = _answers[idx]
            hits = sum(1 for kw in expanded if kw in text)
            if hits >= 2:
                scored.append((idx, hits))
        scored.sort(key=lambda x: -x[1])
        for idx, _ in scored[:3]:
            if _answers[idx] not in related_articles:
                related_articles.append(_answers[idx])
                if len(related_articles) >= 3:
                    break

    # 组合回答
    parts = []

    if best_advise and best_advise_score >= 0.5:
        parts.append(best_advise)

    if related_articles:
        if parts:
            parts.append("\n\n---\n\n**相关法条：**\n")
        else:
            parts.append("**相关法条：**\n")

        for art in related_articles:
            m = re.match(r'根据《(.+?)》第(.+?)条[：:](.*)', art, re.DOTALL)
            if m:
                law_name = m.group(1)
                art_num = m.group(2)
                content = m.group(3).strip()
                if len(content) > 300:
                    content = content[:300] + "……"
                parts.append(f"**《{law_name}》第{art_num}条**：{content}\n")
            else:
                if len(art) > 300:
                    art = art[:300] + "……"
                parts.append(f"{art}\n")

    if not parts and articles:
        idx, score = articles[0]
        return _answers[idx]

    return "".join(parts) if parts else None


# ---- 数学计算 ----

_CHINESE_NUM = {
    '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
    '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
    '十': '10',
}

def _chinese_to_expr(text: str) -> str:
    expr = text.replace('×', '*').replace('÷', '/')
    expr = _parse_power(expr)
    expr = _parse_all_chinese_num(expr)
    expr = expr.replace('加', '+').replace('减', '-').replace('乘', '*').replace('除', '/')
    return expr

def _parse_all_chinese_num(text: str) -> str:
    return re.sub(r'([零一二三四五六七八九十百千万]+)', lambda m: _cn2int(m.group(1)), text)

def _parse_power(text: str) -> str:
    text = text.replace('的平方', '**2').replace('的立方', '**3')
    pat = r'([零一二三四五六七八九十\d]+)的([零一二三四五六七八九十\d]+)次方'
    return re.sub(pat, lambda m: f"{_cn2int(m.group(1))}**{_cn2int(m.group(2))}", text)

def _cn2int(s: str) -> str:
    nm = {'零':0,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10,'百':100,'千':1000,'万':10000}
    r, t = 0, 0
    for c in s:
        if c not in nm:
            if c.isdigit(): t = t * 10 + int(c)
            continue
        v = nm[c]
        if v >= 10000: r, t = (r + t) * v, 0
        elif v >= 100: r, t = r + t * v, 0
        elif v == 10: t = 10 if t == 0 else t * 10
        else: t += v
    return str(r + t) if (r + t) > 0 else s

def _looks_math(text: str) -> bool:
    has_num = any(c.isdigit() or c in _CHINESE_NUM for c in text)
    has_op = any(c in '+-*/^×÷' for c in text) or any(op in text for op in ['加','减','乘','除','的','次方','平方','立方'])
    return has_num and has_op

def _try_calc(text: str) -> str | None:
    m = re.search(r'(?:计算|算一下|算)\s*(.+)', text)
    if m:
        expr = _chinese_to_expr(m.group(1).strip())
    elif _looks_math(text):
        expr = _chinese_to_expr(text.strip())
    else:
        return None
    if not re.match(r'^[\d\s\+\-\*/\.\(\)\^%a-zA-Z_]+$', expr):
        return None
    try:
        safe = expr.replace('^', '**')
        ns = {"math":math,"sqrt":math.sqrt,"pow":math.pow,"sin":math.sin,"cos":math.cos,
              "tan":math.tan,"log":math.log,"log2":math.log2,"log10":math.log10,
              "abs":abs,"round":round,"min":min,"max":max,"pi":math.pi,"e":math.e}
        result = eval(safe, {"__builtins__": {}}, ns)
        return f"计算结果：`{expr.replace('**','^')}` = **{result}**"
    except:
        return None


# ---- 法律关键词检测 ----

_LAW_KEYS = [
    '法', '罪', '判', '刑', '拘留', '逮捕', '起诉', '上诉', '申诉', '诉讼',
    '赔偿', '维权', '合同', '违约', '侵权', '继承', '遗嘱', '离婚', '结婚',
    '抚养', '赡养', '财产', '分割', '借款', '欠款', '债务', '债权',
    '劳动', '工资', '加班', '辞退', '解雇', '试用', '社保', '工伤',
    '消费者', '退货', '假货', '欺诈', '定金', '订金',
    '交通事故', '酒驾', '醉驾', '肇事',
    '知识产权', '著作权', '专利', '商标', '版权',
    '隐私', '个人信息', '网络侵权',
    '正当防卫', '防卫过当', '自首', '缓刑', '取保候审',
    '民事', '刑事', '行政', '诉讼时效', '行为能力',
    '物业', '租赁', '房屋', '买卖',
    '帮信', '寻衅滋事', '诈骗', '盗窃', '故意伤害',
    '打架', '斗殴', '伤人', '欠钱', '借钱', '押金',
    '买房', '购房', '租房', '出租', '试用期', '裁员',
    '被骗', '骗钱', '偷窃', '遗产', '抚养权', '抚养费',
    '车祸', '撞车', '喝酒开车', '加班费',
]

def _is_law_q(text: str) -> bool:
    return any(k in text for k in _LAW_KEYS)


# ---- 主入口 ----

_WEEKDAYS = ["星期一","星期二","星期三","星期四","星期五","星期六","星期日"]

def local_answer(question: str) -> str:
    q = question.strip()

    # 1. 数学计算
    calc = _try_calc(q)
    if calc:
        return calc

    # 2. 时间日期
    now = datetime.datetime.now()
    q_low = q.lower()
    if any(w in q_low for w in ["时间","几点","现在几点","几点钟","当前时间"]):
        return f"现在是 {now.strftime('%H:%M:%S')}。"
    if any(w in q_low for w in ["日期","今天","几号","星期","周几","今天几号"]):
        return f"今天是 {now.strftime('%Y年%m月%d日')}，{_WEEKDAYS[now.weekday()]}。"

    # 3. 问候
    greet = ['你好','您好','hello','hi','嗨','哈喽']
    if any(g in q for g in greet):
        return random.choice(["你好！我是智能问答助手，离线模式下我可以帮你做数学计算和回答法律问题。", "您好！离线模式支持数学计算和法律咨询，请问有什么可以帮您？"])

    # 4. 法律问答（两阶段检索）
    if _is_law_q(q):
        result = _law_rag(q)
        if result:
            return result

    # 5. 帮助提示
    return "离线模式仅支持**数学计算**和**法律问答**。\n\n例如：\n- 数学：`计算 2+3`、`一的二次方`、`15乘8`\n- 法律：`什么是正当防卫`、`离婚财产怎么分割`、`酒驾怎么处罚`\n\n其他问题请切换**在线模式**获取回答。"


# 启动时构建索引
build_index()
