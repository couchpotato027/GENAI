"""
Groq LLM Client
=================
Interfaces with the Groq API to generate structured assessment
analysis using the Qwen3-32B model.

Features:
- Loads API key from environment (never hardcoded)
- Enforces structured JSON output via system prompt
- Temperature 0.2 for deterministic responses
- Retry logic with exponential backoff
- Rule-based fallback if API is unavailable
"""

import os
import json
import time
from typing import Optional


class GroqLLMClient:
    """
    Client for Groq's inference API using Qwen3-32B.
    
    Usage:
        client = GroqLLMClient()  # Reads GROQ_API_KEY from env
        result = client.generate(prompt)  # Returns parsed dict
    """

    MODEL = "qwen/qwen3-32b"
    TEMPERATURE = 0.2
    MAX_TOKENS = 2048
    MAX_RETRIES = 2

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.client = None

        if not self.api_key:
            print("[LLM] WARNING: GROQ_API_KEY not found. LLM features will use fallback mode.")
            return

        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            print(f"[LLM] Groq client initialized with model: {self.MODEL}")
        except ImportError:
            print("[LLM] WARNING: 'groq' package not installed. pip install groq")
        except Exception as e:
            print(f"[LLM] WARNING: Failed to initialize Groq client: {e}")

    @property
    def is_available(self) -> bool:
        """Check if the LLM client is ready to use."""
        return self.client is not None

    def generate(self, prompt: str) -> dict:
        """
        Generate a structured JSON response from the LLM.
        
        Parameters
        ----------
        prompt : The full analysis prompt.
        
        Returns
        -------
        dict : Parsed JSON response with assessment analysis.
        
        Raises
        ------
        RuntimeError : If all retries fail.
        """
        if not self.is_available:
            raise RuntimeError("Groq client not available")

        system_prompt = (
            "You are an expert educational assessment consultant. "
            "You MUST respond with ONLY a valid JSON object. "
            "Do NOT include markdown formatting, code fences, or any text outside the JSON. "
            "Do NOT include ```json or ``` markers. "
            "Your response must be parseable by json.loads() directly."
        )

        last_error = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.TEMPERATURE,
                    max_tokens=self.MAX_TOKENS,
                )

                raw_text = response.choices[0].message.content.strip()
                
                # Handle <think> tags from Qwen models (they sometimes include reasoning)
                if "<think>" in raw_text:
                    # Extract content after </think> tag
                    think_end = raw_text.find("</think>")
                    if think_end != -1:
                        raw_text = raw_text[think_end + len("</think>"):].strip()
                
                # Clean up common LLM formatting issues
                raw_text = self._clean_json_response(raw_text)
                
                # Parse JSON
                result = json.loads(raw_text)

                # Validate required keys exist
                required = ["summary", "difficulty_analysis", "learning_gaps",
                           "question_issues", "recommendations",
                           "pedagogical_references", "ethical_notice"]

                for key in required:
                    if key not in result:
                        result[key] = [] if key in ("learning_gaps", "question_issues",
                                                      "recommendations", "pedagogical_references") else ""

                return result

            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                print(f"[LLM] Attempt {attempt + 1}: {last_error}")
                # Try to extract JSON from the response
                try:
                    result = self._extract_json_from_text(raw_text)
                    if result:
                        return result
                except Exception:
                    pass
                    
            except Exception as e:
                last_error = str(e)
                print(f"[LLM] Attempt {attempt + 1}: API error: {last_error}")

            # Exponential backoff
            if attempt < self.MAX_RETRIES:
                time.sleep(1 * (attempt + 1))

        raise RuntimeError(f"LLM generation failed after {self.MAX_RETRIES + 1} attempts: {last_error}")

    def _clean_json_response(self, text: str) -> str:
        """Remove common LLM formatting artifacts from JSON responses."""
        # Remove markdown code fences
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        return text.strip()

    def _extract_json_from_text(self, text: str) -> Optional[dict]:
        """Try to extract a JSON object from mixed text."""
        # Find the first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None
