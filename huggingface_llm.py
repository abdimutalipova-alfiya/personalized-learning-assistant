from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class HuggingFaceLLMProvider:
    """
    A provider for HuggingFace Large Language Models compatible with CrewAI.
    """

    @classmethod
    def generate_response(cls, prompt: str, **kwargs) -> str:
        """
        Generate a response using EleutherAI/gpt-neox-20bt.

        Args:
            prompt (str): Input prompt.
            **kwargs: Additional generation parameters.

        Returns:
            str: Generated response.
        """
        model_name = "EleutherAI/gpt-neox-20b"
        try:
            # Load the tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,  # Use BF16 precision for faster inference
                trust_remote_code=True,
                device_map="auto",  # Automatically assign model to available devices (e.g., GPU if available)
            )

            # Generate text
            output = model_pipeline(
                prompt,
                max_length=kwargs.get("max_length", 200),  # Limit the maximum token generation
                do_sample=kwargs.get("do_sample", True),
                top_k=kwargs.get("top_k", 10),
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,  # End-of-sequence token
            )

            # Return the generated text
            return output[0]["generated_text"]

        except Exception as e:
            return f"Error generating Falcon-40B-Instruct response: {str(e)}"
