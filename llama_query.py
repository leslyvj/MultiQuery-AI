# llama_query.py
import os
from generator import MultiFallbackGenerator

# Create a singleton generator instance
_generator = None

def get_generator(model_path: str, device: str = "cpu"):
    """Get or create the generator instance"""
    global _generator
    if _generator is None:
        print(f"Initializing generator with model: {model_path}, device: {device}")
        _generator = MultiFallbackGenerator(
            gptq_path=model_path,
            device=device
        )
    return _generator

def generate_answer(prompt: str, model_path: str, max_tokens: int = 256, temperature: float = 0.7, device: str = "cpu") -> str:
    """Generate answer using the multi-fallback generator"""
    try:
        if not os.path.exists(model_path):
            print(f"Warning: LLaMA model not found at {model_path}. Using fallback generation.")
        
        print(f"Generating answer with max_tokens={max_tokens}, temperature={temperature}")
        generator = get_generator(model_path, device)
        answer = generator.generate(prompt, max_tokens=max_tokens)
        
        if not answer or len(answer.strip()) == 0:
            print("Warning: Generated answer is empty!")
            return "I apologize, but I couldn't generate an answer. Please try again."
        
        print(f"Generated answer length: {len(answer)} characters")
        return answer
    except Exception as e:
        print(f"Error in generate_answer: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating answer: {str(e)}"

if __name__ == "__main__":
    print("llama wrapper loaded")
