from abc import ABC, abstractmethod
from typing import List

class PromptStrategy(ABC):
    """Abstract base class for prompting strategies."""

    @abstractmethod
    def build_prompt(self, words: List[str]) -> str:
        pass

class StandardStoryStrategy(PromptStrategy):
    """The baseline: Just ask for a story with the words."""

    def build_prompt(self, words: List[str]) -> str:
        system_msg = "You are a helpful assistant capable of following complex constraints."
        user_msg = f"""Write a detailed story that explicitly includes the following {len(words)} words. 
You must ensure EVERY word from the list appears in the story exactly as written.

LIST OF REQUIRED WORDS:
{', '.join(words)}

STORY:
"""
        return f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"

class IndexedListStrategy(PromptStrategy):
    """Experimental: Asks model to print the word number next to usage."""
    
    def build_prompt(self, words: List[str]) -> str:
        # Create a numbered list string
        numbered_list = "\n".join([f"{i+1}. {w}" for i, w in enumerate(words)])
        
        user_msg = f"""Write a story using these words. 
Whenever you use a word from the list, put its number in brackets, like: apple [1].

WORDS:
{numbered_list}

STORY:
"""
        return f"<|user|>\n{user_msg}\n<|assistant|>\n"