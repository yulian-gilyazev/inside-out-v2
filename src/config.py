import os
from dotenv import load_dotenv

load_dotenv()

VSEGPT_API_KEY = os.getenv('VSE_GPT_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
WANDB_PROJECT_NAME = "inside-out-v2"