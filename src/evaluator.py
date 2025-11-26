from typing import List, Dict, Optional
import re

class Evaluator:
    @staticmethod
    def score_strict(required_words: List[str], generated_text: str) -> Dict:
        """
        Strict exact-match scoring with position tracking.
        Returns a dict containing the score and detailed stats.
        """
        text_lower = generated_text.lower()
        
        # Track each word's status
        word_details = []
        for idx, word in enumerate(required_words):
            word_lower = word.lower()
            # Find all occurrences
            matches = [m.start() for m in re.finditer(re.escape(word_lower), text_lower)]
            
            word_details.append({
                "position": idx,  # Position in the rule list (0-indexed)
                "word": word,
                "found": len(matches) > 0,
                "occurrences": len(matches),
                "positions_in_text": matches  # Character positions in generated text
            })
        
        # Calculate aggregate stats
        passed_words = [wd for wd in word_details if wd["found"]]
        passed_count = len(passed_words)
        total_count = len(required_words)
        score = passed_count / total_count if total_count > 0 else 0.0
        
        return {
            "score": score,
            "passed_count": passed_count,
            "total_count": total_count,
            "missing_words": [wd["word"] for wd in word_details if not wd["found"]],
            "word_details": word_details,  # Full per-word breakdown
            # Aggregated position info
            "followed_positions": [wd["position"] for wd in word_details if wd["found"]],
            "unfollowed_positions": [wd["position"] for wd in word_details if not wd["found"]]
        }