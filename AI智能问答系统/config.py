"""DeepSeek API 配置"""

import os

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-api-key-here")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"