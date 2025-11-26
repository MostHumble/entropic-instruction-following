from typing import List, Dict

class Evaluator:
    @staticmethod
    def score_strict(required_words: List[str], generated_text: str) -> Dict:
        """
        Strict exact-match scoring.
        Returns a dict containing the score and detailed stats.
        """
        text_lower = generated_text.lower()
        
        # Calculate hits
        hits = [w for w in required_words if w.lower() in text_lower]
        passed_count = len(hits)
        total_count = len(required_words)
        
        score = passed_count / total_count if total_count > 0 else 0.0
        
        return {
            "score": score,
            "passed_count": passed_count,
            "total_count": total_count,
            "missing_words": list(set(required_words) - set(hits)) # Useful for debugging
        }