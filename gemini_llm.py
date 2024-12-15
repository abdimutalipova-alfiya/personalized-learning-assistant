import os
import google.generativeai as genai
from litellm import completion

class GeminiLLMProvider:
    """
    A provider for Google's Gemini Large Language Models compatible with CrewAI.
    """

    @classmethod
    def generate_response(cls, prompt: str, **kwargs) -> str:
        """
        Generate a response using Gemini API directly compatible with CrewAI's expected interface.

        Args:
            prompt (str): Input prompt.
            **kwargs: Additional generation parameters.

        Returns:
            str: Generated response.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is missing. Set GEMINI_API_KEY environment variable.")

        try:
            # Use litellm to handle the Gemini API call
            response = completion(
                model="gemini/gemini-pro", 
                api_key=api_key,
                messages=[{"content": prompt, "role": "user"}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1024)
            )

            # Extract the text response
            return response.choices[0].message.content

        except Exception as e:
            return f"Error generating Gemini response: {str(e)}"