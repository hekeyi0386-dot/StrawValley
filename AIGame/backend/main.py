"""
AI 模型 API 接口服务
支持 OpenAI 兼容格式的模型调用（OpenAI、DeepSeek、通义千问、本地模型等）
以及 Hugging Face Inference API
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
from dotenv import load_dotenv

# 优先加载项目根目录的 .env 文件
_env_path = Path(__file__).parent / ".env"
if not _env_path.exists():
    print(
        f"[警告] 未找到 .env 文件({_env_path})，"
        "请复制 .env.example 为 .env 并填入 API Key",
        file=sys.stderr,
    )
load_dotenv(dotenv_path=_env_path)

app = FastAPI(title="AI 模型 API 接口", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 数据模型 ----

class Message(BaseModel):
    role: str        # "system" | "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None          # 不传则使用 .env 中的默认值
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False


class ChatResponse(BaseModel):
    model: str
    content: str
    usage: dict


# ---- 路由 ----

@app.get("/")
def root():
    return {"status": "ok", "message": "AI API 服务运行中"}


@app.get("/models")
def list_models():
    """返回支持的模型列表（从配置读取）"""
    models = os.getenv("AVAILABLE_MODELS", "gpt-4o,gpt-3.5-turbo").split(",")
    return {"models": [m.strip() for m in models]}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    通用聊天接口，兼容所有 OpenAI 格式的模型。
    """
    api_key    = os.getenv("OPENAI_API_KEY", "")
    base_url   = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    default_model = os.getenv("DEFAULT_MODEL", "gpt-4o")

    if not api_key:
        raise HTTPException(status_code=500, detail="未配置 API Key，请检查 .env 文件")

    model = req.model or default_model

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[m.model_dump() for m in req.messages],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"模型调用失败: {str(e)}")

    choice = response.choices[0]
    return ChatResponse(
        model=response.model,
        content=choice.message.content,
        usage=response.usage.model_dump() if response.usage else {},
    )


@app.post("/chat/simple")
def chat_simple(user_input: str, system_prompt: Optional[str] = None):
    """
    简化接口：只传用户文本，自动组装消息列表。
    GET 参数: ?user_input=你好&system_prompt=你是助手
    """
    messages = []
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))
    messages.append(Message(role="user", content=user_input))

    req = ChatRequest(messages=messages)
    return chat(req)


# ---- Hugging Face 路由 ----

# 推荐的免费开源对话模型
HF_DEFAULT_MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",     # 微软 Phi-3，4K 上下文
    "mistralai/Mistral-7B-Instruct-v0.3",   # Mistral 7B
    "HuggingFaceH4/zephyr-7b-beta",         # Zephyr 7B
    "Qwen/Qwen2.5-7B-Instruct",             # 通义千问 7B（中文友好）
    "meta-llama/Meta-Llama-3-8B-Instruct",  # Llama3 8B（需申请权限）
]


@app.get("/hf/models")
def hf_models():
    """返回推荐的 Hugging Face 免费模型列表"""
    return {"models": HF_DEFAULT_MODELS}


@app.post("/hf/chat", response_model=ChatResponse)
def hf_chat(req: ChatRequest):
    """
    通过 Hugging Face Router(OpenAI 兼容接口)调用开源模型。
    默认模型: Qwen2.5-7B-Instruct（中文友好）
    在 .env 中设置 HF_TOKEN、HF_DEFAULT_MODEL、HF_BASE_URL 来自定义。
    """
    hf_token = os.getenv("HF_TOKEN", "")
    hf_base_url = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")
    default_model = os.getenv("HF_DEFAULT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    model = req.model or default_model

    if not hf_token:
        raise HTTPException(
            status_code=500,
            detail="未配置 HF_TOKEN，请在 .env 中添加。免费申请: https://huggingface.co/settings/tokens",
        )

    client = OpenAI(api_key=hf_token, base_url=hf_base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[m.model_dump() for m in req.messages],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HF 模型调用失败: {str(e)}")

    choice = response.choices[0]
    usage = {}
    if response.usage:
        usage = response.usage.model_dump()
    return ChatResponse(
        model=model,
        content=choice.message.content,
        usage=usage,
    )


@app.post("/hf/chat/simple")
def hf_chat_simple(user_input: str, model: Optional[str] = None, system_prompt: Optional[str] = None):
    """
    HF 简化接口：只传用户文字即可。
    ?user_input=你好&model=Qwen/Qwen2.5-7B-Instruct
    """
    messages = []
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))
    messages.append(Message(role="user", content=user_input))

    req = ChatRequest(messages=messages, model=model)
    return hf_chat(req)


# ---- Stardew Valley AI 规划接口 ----

class PlayerProfile(BaseModel):
    level_farming: int = 0
    level_mining: int = 0
    level_foraging: int = 0
    level_fishing: int = 0
    gold: int = 0
    energy: int = 100
    season: str = "spring"
    day: int = 1
    weather: str = "sunny"


class FarmProgress(BaseModel):
    house_level: int = 1
    coop_built: bool = False
    barn_built: bool = False
    silo_built: bool = False
    furnace_count: int = 0
    chest_count: int = 0
    tool_level: str = "basic"


class InventoryItem(BaseModel):
    name: str
    count: int


class CropItem(BaseModel):
    name: str
    count: int
    days_to_harvest: int


class PlanRequest(BaseModel):
    player_profile: PlayerProfile
    farm_progress: FarmProgress
    inventory: List[InventoryItem] = []
    crops: List[CropItem] = []
    goals: List[str] = []
    user_intent: str = ""


class PlanResponse(BaseModel):
    priority_summary: str
    today_plan: List[str]
    build_priority: str
    sell_keep_process: List[str]
    next_3_day_focus: List[str]
    reason: str
    risk_warning: str
    risk: str


_SEASON_ZH = {"spring": "春季", "summer": "夏季", "fall": "秋季", "winter": "冬季"}
_WEATHER_ZH = {"sunny": "晴天", "rainy": "下雨（无需浇水）", "windy": "多风", "festival": "节日"}
_TOOL_ZH = {"basic": "基础级", "copper": "铜级", "steel": "铁级", "gold": "金级"}
_GOAL_ZH = {
    "maximize_profit": "收益最大化",
    "upgrade_fast": "尽快升级",
    "prepare_buildings": "尽快建造",
    "save_energy": "节省体力",
    "stable_beginner_path": "稳健新手路线",
}


def _build_plan_prompt(req: PlanRequest) -> str:
    p = req.player_profile
    f = req.farm_progress
    inv = {item.name: item.count for item in req.inventory}
    crop_lines = "；".join(
        f"{c.name} x{c.count}（还有 {c.days_to_harvest} 天收获）" for c in req.crops
    ) or "无"
    goals_str = "、".join(_GOAL_ZH.get(g, g) for g in req.goals) or "无明确目标"
    buildings = []
    if f.coop_built:   buildings.append("鸡舍")
    if f.barn_built:   buildings.append("畜棚")
    if f.silo_built:   buildings.append("筒仓")
    buildings_str = "、".join(buildings) if buildings else "无"

    user_msg = f"""
你是《星露谷物语》（Stardew Valley）专业攻略助手，擅长帮新手规划最优日程。

## 当前农场状态
- 季节/日期/天气：{_SEASON_ZH.get(p.season, p.season)} 第 {p.day} 天 / {_WEATHER_ZH.get(p.weather, p.weather)}
- 金币：{p.gold}g　体力：{p.energy}/100
- 技能等级：耕种 {p.level_farming}、采矿 {p.level_mining}、觅食 {p.level_foraging}、钓鱼 {p.level_fishing}
- 房屋等级：{f.house_level}　工具平均等级：{_TOOL_ZH.get(f.tool_level, f.tool_level)}
- 熔炉：{f.furnace_count} 个　箱子：{f.chest_count} 个
- 已建造：{buildings_str}
- 库存：木头 {inv.get('wood',0)}、石头 {inv.get('stone',0)}、纤维 {inv.get('fiber',0)}、黏土 {inv.get('clay',0)}、铜矿 {inv.get('copper_ore',0)}、煤炭 {inv.get('coal',0)}、铜锭 {inv.get('copper_bar',0)}、食物 {inv.get('food',0)}
- 作物：{crop_lines}
- 目标偏好：{goals_str}
- 玩家描述：{req.user_intent}

## 输出要求
请严格以 JSON 格式输出，不要有任何多余文字，格式如下：
{{
  "priority_summary": "一段话总结今日最高优先级行动",
  "today_plan": ["步骤1", "步骤2", "步骤3", "步骤4", "步骤5"],
  "build_priority": "下一个最该建造的建筑及原因",
  "sell_keep_process": ["卖/留/加工建议1", "建议2", "建议3"],
  "next_3_day_focus": ["day1重点", "day2重点", "day3重点"],
  "reason": "综合分析为何这样安排，考虑了哪些因素",
  "risk_warning": "今日需要注意的风险或容易踩的坑",
  "risk": "low 或 medium 或 high"
}}
""".strip()
    return user_msg


def _parse_plan_json(raw: str) -> dict:
    """从 AI 回复中提取 JSON，兼容前后有多余文字的情况。"""
    # 尝试直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 提取第一个 {...} 块
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _build_local_fallback_plan(req: PlanRequest, warning: str = "") -> dict:
    """当外部模型不可用时，返回可用的本地兜底规划，保证前端交互不断流。"""
    p = req.player_profile
    f = req.farm_progress
    inv = {item.name: item.count for item in req.inventory}

    need_water = p.weather != "rainy"
    has_low_energy = p.energy < 45
    wood = inv.get("wood", 0)
    stone = inv.get("stone", 0)
    copper_ore = inv.get("copper_ore", 0)
    coal = inv.get("coal", 0)

    build_priority = "先补资源（木头/石头）"
    if not f.coop_built and wood >= 300 and stone >= 100 and p.gold >= 4000:
        build_priority = "优先建鸡舍：资源和金币已接近或达到门槛，能尽早形成稳定收益"
    elif not f.silo_built and wood >= 100 and stone >= 100 and p.gold >= 100:
        build_priority = "优先建筒仓：为后续养殖准备干草体系，降低冬季压力"
    elif copper_ore >= 20 and coal >= 5:
        build_priority = "优先准备工具升级：先烧铜锭，再安排次日升级，提高长期效率"

    today_plan = []
    if need_water:
        today_plan.append("早晨先浇水并收成熟作物，避免漏浇导致生长延迟")
    else:
        today_plan.append("下雨天免浇水，直接把体力投入采矿或砍树")

    today_plan.append("10:00前整理背包，只带食物、工具和目标资源，减少跑图与清包成本")
    if has_low_energy:
        today_plan.append("中午优先做低体力收益行为（觅食/钓鱼），体力低于20立即回家")
    else:
        today_plan.append("中午到傍晚主攻矿洞铜层，目标铜矿>=25、煤炭>=8")
    today_plan.append("傍晚回农场处理矿石入炉，木石不足则补采到次日建造阈值")
    today_plan.append("睡前复盘：确认明日天气和商店计划，预留金币不冲动消费")

    sell_keep_process = [
        "普通品质作物可卖换现金流；高价值或任务相关作物保留1-2组",
        "木头/石头优先留作建造，不建议前期大量出售",
        "铜矿+煤炭优先加工成铜锭，提升后续工具升级节奏",
    ]

    next_3_day_focus = [
        "Day1：完成当日浇水与资源补给，确保铜矿与木石储备达标",
        "Day2：推进一次关键建造或工具升级，形成效率拐点",
        "Day3：围绕收益循环（种植-收获-再投资）稳定扩张",
    ]

    risk = "medium"
    risk_warning = "避免体力透支和资金见底；前期最怕同时断体力、断金币、断资源。"
    if has_low_energy:
        risk = "high"
        risk_warning = "当前体力偏低，若强行下矿容易导致次日效率崩盘，建议保留恢复食物。"

    if warning:
        risk_warning = f"{risk_warning}（系统提示：{warning}）"

    return {
        "priority_summary": "今天先保核心循环：作物维护 + 关键资源补齐 + 晚间加工，为明后天升级或建造铺路。",
        "today_plan": today_plan,
        "build_priority": build_priority,
        "sell_keep_process": sell_keep_process,
        "next_3_day_focus": next_3_day_focus,
        "reason": "本方案根据天气、体力、金币和关键材料储备做保守优化，优先保证可持续发展与新手容错。",
        "risk_warning": risk_warning,
        "risk": risk,
    }


@app.post("/plan", response_model=PlanResponse)
def plan(req: PlanRequest):
    """
    Stardew Valley 农场规划接口。
    接收前端游戏状态，返回 AI 生成的今日行动计划。
    优先使用 HF_TOKEN，若未配置则回落到 OPENAI_API_KEY。
    """
    prompt = _build_plan_prompt(req)
    hf_token   = os.getenv("HF_TOKEN", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    raw_content = ""
    fail_reason = ""

    # --- 优先走 Hugging Face Router ---
    if hf_token:
        hf_base_url = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")
        hf_model = os.getenv("HF_DEFAULT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        client = OpenAI(api_key=hf_token, base_url=hf_base_url)
        try:
            resp = client.chat.completions.create(
                model=hf_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1200,
            )
            raw_content = resp.choices[0].message.content
        except Exception as e:
            fail_reason = f"HF 模型调用失败: {e}"
            # 如果 HF 不可用，自动回落到 OpenAI 兼容接口，减少服务不可用情况
            if openai_key:
                base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
                model = os.getenv("DEFAULT_MODEL", "gpt-4o")
                client = OpenAI(api_key=openai_key, base_url=base_url)
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4,
                        max_tokens=1200,
                    )
                    raw_content = resp.choices[0].message.content
                except Exception as openai_e:
                    fail_reason = f"{fail_reason}；OpenAI 回落也失败: {openai_e}"

    # --- 回落到 OpenAI 兼容接口 ---
    elif openai_key:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model    = os.getenv("DEFAULT_MODEL", "gpt-4o")
        client   = OpenAI(api_key=openai_key, base_url=base_url)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1200,
            )
            raw_content = resp.choices[0].message.content
        except Exception as e:
            fail_reason = f"OpenAI 模型调用失败: {e}"

    # 如果没有可用密钥，直接返回本地兜底结果，保证前端可用。
    elif not hf_token and not openai_key:
        data = _build_local_fallback_plan(req, "未配置 HF_TOKEN 或 OPENAI_API_KEY，已使用本地规划")
        return PlanResponse(**data)

    data = _parse_plan_json(raw_content)
    if not data:
        # 外部模型返回不可解析时也返回本地方案，避免前端报错中断。
        warn = fail_reason or f"AI 返回内容无法解析为 JSON：{raw_content[:120]}"
        data = _build_local_fallback_plan(req, warn)

    return PlanResponse(
        priority_summary  = data.get("priority_summary", ""),
        today_plan        = data.get("today_plan", []),
        build_priority    = data.get("build_priority", ""),
        sell_keep_process = data.get("sell_keep_process", []),
        next_3_day_focus  = data.get("next_3_day_focus", []),
        reason            = data.get("reason", ""),
        risk_warning      = data.get("risk_warning", ""),
        risk              = data.get("risk", "medium"),
    )


# ---- 启动 ----

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    reload_mode = os.getenv("RELOAD", "false").strip().lower() == "true"
    uvicorn.run(app, host="0.0.0.0", port=port, reload=reload_mode)
