# generator.py
"""
Multi-fallback generator for multimodal RAG.
Supports Ollama (Phi-3), Transformers models, and seq2seq fallback.
"""
import os
from typing import Optional


class MultiFallbackGenerator:
    def __init__(self, gptq_path: Optional[str] = None, offline_seq2seq: Optional[str] = None, device: str = "cpu"):
        self.gptq_path = gptq_path
        self.offline_seq2seq = offline_seq2seq
        self.device = device
        self._backend = None

    def _try_load_ollama(self):
        """Try loading model via Ollama (fastest and most efficient)"""
        try:
            print("Attempting to load model via Ollama...")
            from langchain_ollama import OllamaLLM
            
            # Check if model path looks like an Ollama model name
            model_name = self.gptq_path
            if not model_name or "/" in model_name:
                # Convert HuggingFace format to Ollama format
                if "Phi-3" in str(model_name) or "phi3" in str(model_name).lower():
                    model_name = "phi3"
                else:
                    print(f"Model path '{model_name}' doesn't look like an Ollama model")
                    return None
            
            print(f"Loading Ollama model: {model_name}")
            # Store model name for later use
            self.model_name = model_name
            # Optimized for ChatGPT-style structured responses with complete, well-formatted answers
            llm = OllamaLLM(
                model=model_name,
                temperature=0.5,  # Balanced for natural, structured responses
                num_predict=2048,  # Increased to 2048 for complete answers with ALL sections (Summary + Key Points + Steps + Insights + Takeaway)
                num_ctx=4096,  # Good context window
                top_k=50,
                top_p=0.92,  # Slightly higher for more natural language
                repeat_penalty=1.1,  # Lower penalty for more natural repetition
                num_gpu=1,  # Use GPU acceleration
                num_thread=4,
                timeout=300  # 5 minutes - allow more time for longer, structured responses
            )
            
            # Test if Ollama is running and model is available
            try:
                test_response = llm.invoke("test")
                print(f"Successfully loaded Ollama model: {model_name}")
                return ("ollama", None, llm)
            except Exception as test_error:
                print(f"Ollama model test failed: {test_error}")
                print("Make sure Ollama is running: 'ollama serve'")
                print(f"And model is downloaded: 'ollama pull {model_name}'")
                return None
                
        except ImportError:
            print("langchain-ollama not installed")
            return None
        except Exception as e:
            print(f"Failed to load Ollama: {e}")
            return None

    def _try_load_auto_gptq(self):
        """Try loading causal LM model (Phi-3, LLaMA, etc.) via Transformers"""
        try:
            print("Attempting to load causal language model via Transformers...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_dir = self.gptq_path
            if model_dir is None:
                print("No model path specified")
                return None
            
            # Check if it's a local path or HuggingFace model ID
            is_local = os.path.exists(model_dir)
            
            # Check if CUDA is available
            if not torch.cuda.is_available():
                print("CUDA not available. Using CPU (will be slower)...")
                device_map = "cpu"
                dtype = torch.float32
            else:
                print(f"CUDA available: {torch.cuda.get_device_name(0)}")
                device_map = "auto"
                dtype = torch.float16
            
            print(f"Loading tokenizer from {model_dir}")
            tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
            
            # Set padding token if not set
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            
            print(f"Loading model from {model_dir} (dtype: {dtype}, device: {device_map})...")
            if not is_local:
                print("Downloading from HuggingFace (this may take a few minutes on first run)...")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map=device_map,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager"
            )
            
            print(f"Successfully loaded {model_dir}!")
            
            return ("causal_lm", tok, model)
                
        except Exception as e:
            print(f"Failed to load causal LM model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _try_load_seq2seq(self):
        try:
            print("Attempting to load seq2seq fallback model...")
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import os
            # Check for fallback model in environment
            model_name = os.getenv("FALLBACK_MODEL", self.offline_seq2seq or "google/flan-t5-base")
            print(f"Loading {model_name}...")
            tok = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            print(f"Successfully loaded seq2seq model: {model_name}")
            return ("seq2seq", tok, model)
        except Exception as e:
            print(f"Failed to load seq2seq: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_backend(self):
        if self._backend is not None:
            return self._backend
        print("Initializing backend...")
        
        # Try Ollama first (fastest and most efficient)
        o = self._try_load_ollama()
        if o:
            print(f"Using {o[0]} backend")
            self._backend = o
            return self._backend
        
        # Try Transformers models
        g = self._try_load_auto_gptq()
        if g:
            print(f"Using {g[0]} backend")
            self._backend = g
            return self._backend
        s = self._try_load_seq2seq()
        if s:
            print("Using seq2seq backend")
            self._backend = s
            return self._backend
        print("Using extractive fallback backend")
        self._backend = ("extractive", None, None)
        return self._backend

    def generate(self, prompt: str, max_tokens: int = 256):
        backend = self._get_backend()
        kind, tok, model = backend
        print(f"Generating with backend: {kind}, max_tokens: {max_tokens}")
        try:
            if kind == "ollama":
                print(f"Generating with Ollama (length: {len(prompt)} chars)...")
                # Create a fresh Ollama instance with the correct max_tokens for this generation
                from langchain_ollama import OllamaLLM
                fresh_model = OllamaLLM(
                    model=self.model_name,
                    temperature=0.5,
                    num_predict=max_tokens,  # Use the passed max_tokens value
                    num_ctx=4096,
                    top_k=50,
                    top_p=0.92,
                    repeat_penalty=1.1,
                    num_gpu=1,
                    num_thread=4,
                    timeout=300
                )
                response = fresh_model.invoke(prompt)
                print(f"Generated {len(response)} characters")
                
                # Clean up the output
                generated_text = response.strip()
                
                # Ensure complete sentences - remove trailing incomplete sentences
                if generated_text and not generated_text[-1] in '.!?':
                    # Find the last sentence-ending punctuation
                    last_period = max(
                        generated_text.rfind('.'),
                        generated_text.rfind('!'),
                        generated_text.rfind('?')
                    )
                    # Only trim if we have substantial content before the last punctuation
                    if last_period > len(generated_text) * 0.4:  # Keep if >40% of content
                        generated_text = generated_text[:last_period + 1]
                        print(f"Trimmed incomplete sentence. Final length: {len(generated_text)} characters")
                
                return generated_text
            
            elif kind == "causal_lm":
                import torch
                print(f"Tokenizing prompt (length: {len(prompt)} chars)...")
                
                # Move model to GPU if available and not already there
                device = next(model.parameters()).device
                inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
                input_length = inputs.input_ids.shape[1]
                print(f"Input tokens: {input_length}, generating up to {max_tokens} new tokens on {device}...")
                
                # Better generation parameters for structured responses
                with torch.no_grad():
                    out = model.generate(
                        **inputs, 
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1,
                        pad_token_id=tok.eos_token_id,
                        eos_token_id=tok.eos_token_id
                    )
                
                # Decode only the new tokens (skip the prompt)
                generated_text = tok.decode(out[0][input_length:], skip_special_tokens=True)
                
                # Clean up the output
                generated_text = generated_text.strip()
                
                # Ensure complete sentences - remove trailing incomplete sentences
                if generated_text and not generated_text[-1] in '.!?':
                    # Find the last sentence-ending punctuation
                    last_period = max(
                        generated_text.rfind('.'),
                        generated_text.rfind('!'),
                        generated_text.rfind('?')
                    )
                    # Only trim if we have substantial content before the last punctuation
                    if last_period > len(generated_text) * 0.4:  # Keep if >40% of content
                        generated_text = generated_text[:last_period + 1]
                        print(f"Trimmed incomplete sentence. Final length: {len(generated_text)} characters")
                
                print(f"Generated {len(generated_text)} characters")
                return generated_text
                
            elif kind == "seq2seq":
                import torch
                print(f"Tokenizing prompt for seq2seq (length: {len(prompt)} chars)...")
                
                # Move model to GPU if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                
                # Tokenize and move to device
                inputs = tok(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
                print(f"Input tokens: {inputs.input_ids.shape[1]}, generating {max_tokens} tokens on {device}...")
                
                # Generate with better parameters
                out = model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    length_penalty=1.0,
                    num_beams=1
                )
                generated_text = tok.decode(out[0], skip_special_tokens=True)
                
                # Clean up the output
                generated_text = generated_text.strip()
                
                # Ensure complete sentences - remove trailing incomplete sentences
                if generated_text and not generated_text[-1] in '.!?':
                    # Find the last sentence-ending punctuation
                    last_period = max(
                        generated_text.rfind('.'),
                        generated_text.rfind('!'),
                        generated_text.rfind('?')
                    )
                    # Only trim if we have substantial content before the last punctuation
                    if last_period > len(generated_text) * 0.4:  # Keep if >40% of content
                        generated_text = generated_text[:last_period + 1]
                        print(f"Trimmed incomplete sentence. Final length: {len(generated_text)} characters")
                
                print(f"Generated {len(generated_text)} characters")
                return generated_text
            else:
                # very simple extractive fallback: return a note
                return "I cannot generate an answer at this time. Please ensure a language model is properly loaded."
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating answer: {str(e)}"
