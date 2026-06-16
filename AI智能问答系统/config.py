"""DeepSeek API 配置"""

from dotenv import load_dotenv
import os


def _load_dotenv():
    _dir = os.path.dirname(os.path.abspath(__file__))
    _env = os.path.join(_dir, ".env")
    if os.path.exists(_env):
        load_dotenv(_env)
        print(f"[config] 已加载 {_env}")
    else:
        print(f"[config] 未找到 .env，使用环境变量或默认值")


_load_dotenv()

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-api-key-here")
BASE_URL = os.environ.get("BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("MODEL", "deepseek-chat")
