"""
SmartShield — ExecuTorch Llama Setup Guide & Loader
Instructions and code for deploying Llama 3.2 1B on ARM Android via ExecuTorch.

This file serves as both documentation and the Stage 3 loader.
"""

# =============================================================================
# SETUP INSTRUCTIONS (Run these on your development machine)
# =============================================================================
#
# Step 1: Install ExecuTorch
# --------------------------
# Follow: https://learn.arm.com/courses/executorch-on-arm/
#
#   git clone https://github.com/pytorch/executorch.git
#   cd executorch
#   git checkout v0.4.0   # or latest stable
#   pip install .
#   # Install XNNPACK backend for ARM:
#   pip install executorch[xnnpack]
#
#
# Step 2: Download Llama 3.2 1B Instruct
# ----------------------------------------
# Request access from Meta: https://llama.meta.com/
# Download: Llama-3.2-1B-Instruct
#
#   # Or via Hugging Face:
#   pip install huggingface_hub
#   huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir models/llama-3.2-1b
#
#
# Step 3: Export to ExecuTorch (.pte) with 4-bit quantization
# -----------------------------------------------------------
#   cd executorch
#
#   python -m examples.models.llama.export_llama \
#       --model "meta-llama/Llama-3.2-1B-Instruct" \
#       --checkpoint models/llama-3.2-1b/consolidated.00.pth \
#       --params models/llama-3.2-1b/params.json \
#       -kv \
#       --use_sdpa_with_kv_cache \
#       -X \
#       --xnnpack-extended-ops \
#       -qmode 8da4w \
#       --group_size 128 \
#       -d fp32 \
#       --max_seq_length 512 \
#       --output_name smartshield_llama_1b_q4.pte
#
#   # This creates: smartshield_llama_1b_q4.pte (~700MB)
#   # Copy to: SmartShield/models/smartshield_llama_1b_q4.pte
#
#
# Step 4: Copy tokenizer
# -----------------------
#   cp models/llama-3.2-1b/tokenizer.model SmartShield/models/llama_tokenizer.model
#
#
# Step 5: Test on device
# -----------------------
#   # Push to Android phone:
#   adb push models/smartshield_llama_1b_q4.pte /data/local/tmp/
#   adb push models/llama_tokenizer.model /data/local/tmp/
#
#   # Or on Raspberry Pi, just copy to models/ directory
#
# =============================================================================

import os
import time
from typing import Optional


class LlamaExplainer:
    """
    Stage 3: Llama LLM Explainer via ExecuTorch.
    
    Generates human-readable explanations of why a message is suspicious,
    grounded entirely in the detector's structured findings.
    """

    def __init__(self, model_path: str, tokenizer_path: str):
        """
        Load Llama model via ExecuTorch.
        
        Args:
            model_path: Path to .pte file (ExecuTorch exported model)
            tokenizer_path: Path to tokenizer.model (SentencePiece)
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self):
        """Load the ExecuTorch model and tokenizer."""
        try:
            # ExecuTorch Python API
            from executorch.extension.pybindings import portable_lib as exec_lib
            from executorch.extension.llm.tokenizer.tokenizer import Tokenizer
            
            print(f"[LLM] Loading ExecuTorch model: {self.model_path}")
            start = time.time()
            
            self.model = exec_lib.ExecuTorchModule(self.model_path)
            self.tokenizer = Tokenizer(self.tokenizer_path)
            
            elapsed = time.time() - start
            print(f"[LLM] Model loaded in {elapsed:.1f}s")
            
        except ImportError:
            print("[LLM] ExecuTorch not installed. Using template-based fallback.")
            print("[LLM] Install ExecuTorch: pip install executorch")
            self.model = None
        except Exception as e:
            print(f"[LLM] Error loading model: {e}")
            self.model = None

    def explain(self, message: str, category: str, features: list,
                risk_score: float, max_tokens: int = 150) -> Optional[str]:
        """
        Generate explanation for a flagged message.
        
        Args:
            message: Original SMS text
            category: Detected threat category
            features: List of detected features
            risk_score: Final risk score (0.0-1.0)
            max_tokens: Max tokens to generate
            
        Returns:
            Human-readable explanation string, or None if model unavailable.
        """
        if self.model is None:
            return None
        
        prompt = self._build_prompt(message, category, features, risk_score)
        
        # Tokenize
        tokens = self.tokenizer.encode(prompt)
        
        # Generate
        start = time.time()
        generated_tokens = []
        time_to_first_token = None
        
        for i in range(max_tokens):
            # Forward pass
            output = self.model.forward(tokens)
            next_token = output.argmax(-1)[-1].item()
            
            if time_to_first_token is None:
                time_to_first_token = time.time() - start
            
            # EOS check
            if next_token == self.tokenizer.eos_id:
                break
            
            generated_tokens.append(next_token)
            tokens = [next_token]
        
        total_time = time.time() - start
        tokens_per_sec = len(generated_tokens) / total_time if total_time > 0 else 0
        
        result = self.tokenizer.decode(generated_tokens)
        
        print(f"[LLM] Time to first token: {time_to_first_token:.2f}s")
        print(f"[LLM] Tokens/sec: {tokens_per_sec:.1f}")
        print(f"[LLM] Total tokens: {len(generated_tokens)}")
        
        return result.strip()

    def _build_prompt(self, message: str, category: str, features: list,
                      risk_score: float) -> str:
        """Build structured prompt for Llama."""
        features_str = ", ".join(features[:5]) if features else "general suspicious patterns"
        
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a cybersecurity assistant running on-device. A spam detector analyzed an SMS and produced the following findings. Explain in 2-3 simple sentences why this is suspicious and suggest one safe action.

Rules:
- Base your explanation ONLY on the detector findings below
- Do NOT invent evidence not found by the detector
- Do NOT instruct the user to click any link or call any number from the message
- Suggest a safe alternative action<|eot_id|><|start_header_id|>user<|end_header_id|>

Risk Score: {risk_score:.0%}
Category: {category}
Detected Features: {features_str}

SMS: "{message[:200]}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    @property
    def is_available(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None


# Quick test
if __name__ == "__main__":
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models", "smartshield_llama_1b_q4.pte"
    )
    tokenizer_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models", "llama_tokenizer.model"
    )
    
    if os.path.exists(model_path):
        explainer = LlamaExplainer(model_path, tokenizer_path)
        result = explainer.explain(
            message="Dear SBI customer, your KYC has expired. Click: http://sbi-kyc.xyz",
            category="fake_kyc",
            features=["fake_kyc_pattern", "suspicious_url", "bank_impersonation"],
            risk_score=0.9
        )
        if result:
            print(f"\nExplanation: {result}")
    else:
        print(f"Model not found at {model_path}")
        print("Follow the setup instructions in this file to export the model.")
        print("\nTemplate-based fallback will be used until model is available.")
