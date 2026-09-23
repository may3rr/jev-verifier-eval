"""Jev（Typesafe AI 的 System One 模型）最小调用示例。

Jev 是什么
    Jev 不是聊天大模型，不逐 token 生成文本。它属于 Typesafe 所说的 System One 模型：
    你给它一份结构化数据（state）和一组带类型的问题（questions），它一次性返回
    每个问题的类型化答案，而不是一段自然语言。

三种问题类型
    1. 是非题：返回 true 的校准概率（本项目用的 "noul" 类型即属于这一类）
    2. 单选题：从你预先定义的选项里选一个
    3. 量表题：在你定义的刻度上给出一个位置

特点
    - 延迟约 100 毫秒，比前沿 LLM 快约 40 到 200 倍，输出 token 免费
    - 输出是概率而不是文本，所以不会编造内容，可直接算 Brier、ECE 等校准指标
    - 不擅长开放式生成、改写、总结，只适合判断、分类、打分

对本课题的用途
    适合当作低成本判官，批量判断候选答案是否保持了证据的范围边界、
    是否跨来源拼出了没有证据支持的复合断言、是否应当拒答，
    再和人工金标做校准比较。

调用方式
    POST https://api.typesafe.ai/v1/systemone
    请求头  Authorization: Bearer <TYPESAFE_API_KEY>
    请求体  {"model": "jev-latest", "state": {...}, "questions": {...}}

密钥注入
    密钥不写进本文件。运行时先读环境变量 TYPESAFE_API_KEY，
    读不到再去 ~/.zshrc 里解析 export 行，两处都没有则报错退出。

状态
    截至 2026 年 9 月中旬为早期访问，需要在候补名单里放行后才能调用。
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = os.environ.get("TYPESAFE_SYSTEMONE_URL", "https://api.typesafe.ai/v1/systemone")
DEFAULT_MODEL = "jev-latest"


def _from_zshrc(name: str) -> str | None:
    zshrc = Path.home() / ".zshrc"
    if not zshrc.exists():
        return None
    pattern = re.compile(rf"^\s*export\s+{re.escape(name)}=(['\"]?)(.+?)\1\s*$")
    value = None
    for line in zshrc.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            value = match.group(2)
    return value


def load_config() -> tuple[str, str]:
    key = os.environ.get("TYPESAFE_API_KEY") or _from_zshrc("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("找不到 TYPESAFE_API_KEY，请先在环境变量或 ~/.zshrc 里配置")
    model = os.environ.get("JEV_MODEL") or _from_zshrc("JEV_MODEL") or DEFAULT_MODEL
    return key, model


def ask_jev(state: dict, questions: dict, key: str, model: str, timeout: int = 60) -> dict:
    body = json.dumps({"model": model, "state": state, "questions": questions}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:500]}")


if __name__ == "__main__":
    api_key, model_name = load_config()
    print(f"model={model_name}  key=已加载（{api_key[:12]}…）")

    # 示例：判断候选答案有没有把某一地区的事实扩大成更广的结论
    demo_state = {
        "question": "非遗传承人认定制度在全国范围内有什么规定？",
        "candidate_text": "全国所有省份都要求传承人必须年满 60 周岁。",
        "records": [{"source": "某省文旅厅通知", "text": "本省申报传承人一般应年满 60 周岁。"}],
    }
    demo_questions = {
        "scope_collapsed": {
            "type": "noul",
            "instructions": "候选答案是否把单一来源的局部事实扩大成了更广范围的事实？",
            "criteria": {
                "true": "范围边界被扩大或丢失。",
                "false": "没有这种扩大或丢失。",
            },
        }
    }
    print(json.dumps(ask_jev(demo_state, demo_questions, api_key, model_name), ensure_ascii=False, indent=2))
