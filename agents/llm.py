"""공용 Chat 모델 인스턴스."""

from langchain_openai import ChatOpenAI

from config import LLM_MODEL, LLM_TEMPERATURE


def get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
