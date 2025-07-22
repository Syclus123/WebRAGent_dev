from .openai import GPTGenerator, GPTGeneratorWithJSON
from .claude import ClaudeGenerator
from .gemini import GeminiGenerator
from .togetherai import TogetherAIGenerator
from .operator import OperatorGenerator, OperatorGeneratorWithJSON


def create_llm_instance(model, json_mode=False, all_json_models=None):
    # OpenAI Operator support - check for both "operator" and the actual model name
    if "operator" in model or "computer-use-preview" in model:
        if json_mode:
            if model in all_json_models:
                return OperatorGeneratorWithJSON(model)
            else:
                raise ValueError("The operator model does not support JSON mode.")
        else:
            return OperatorGenerator(model)
    # OpenAI GPT models
    elif any(keyword in model for keyword in ["gpt", "o1", "o3-mini", "o4-mini"]):
        if json_mode:
            if model in all_json_models:
                return GPTGeneratorWithJSON(model)
            else:
                raise ValueError("The text model does not support JSON mode.")
        else:
            return GPTGenerator(model)
    elif "claude" in model:
        if json_mode:
            raise ValueError("Claude does not support JSON mode.")
        else:
            return ClaudeGenerator(model)
    elif "gemini" in model:
        if json_mode:
            raise ValueError("Gemini does not support JSON mode.")
        else:
            return GeminiGenerator(model)
    else:
        if json_mode:
            raise ValueError("TogetherAI does not support JSON mode.")
        else:
            return TogetherAIGenerator(model)

async def semantic_match_llm_request(messages: list = None):
    GPT35 = GPTGenerator(model="gpt-3.5-turbo")
    return await GPT35.request(messages)