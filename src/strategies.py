from abc import ABC, abstractmethod
from typing import List, Optional
from transformers import AutoTokenizer

class PromptStrategy(ABC):
    """Abstract base class for prompting strategies."""
    
    def __init__(self, model_name: str):
        """Initialize with model name to load correct tokenizer."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    @abstractmethod
    def build_prompt(self, words: List[str]) -> str:
        pass
    
    def format_with_chat_template(self, messages: List[dict]) -> str:
        """Use model's official chat template"""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


class StandardStoryStrategy(PromptStrategy):
    """Baseline: Ask for a story with the words using proper chat format."""

    def build_prompt(self, words: List[str]) -> str:
        system_msg = "You are a helpful assistant capable of following complex constraints."
        
        user_msg = f"""Write a detailed story that explicitly includes the following {len(words)} words. 
You must ensure EVERY word from the list appears in the story exactly as written.

LIST OF REQUIRED WORDS:
{', '.join(words)}"""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        return self.format_with_chat_template(messages)


class IndexedListStrategy(PromptStrategy):
    """Experimental: Asks model to print the word number next to usage."""
    
    def build_prompt(self, words: List[str]) -> str:
        numbered_list = "\n".join([f"{i+1}. {word}" for i, word in enumerate(words)])
        
        user_msg = f"""Write a story using these words. 
Whenever you use a word from the list, put its number in brackets, like: apple [1].

WORDS:
{numbered_list}"""
        
        messages = [
            {"role": "user", "content": user_msg}
        ]
        
        return self.format_with_chat_template(messages)