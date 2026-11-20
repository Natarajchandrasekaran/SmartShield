"""
SmartShield — Main Pipeline
3-Stage Hierarchical SMS Scam Detection System

Stage 1: Rule Engine (regex, <1ms)
Stage 2: ML Classifier (TinyBERT/ONNX, ~5-10ms)
Stage 3: LLM Explainer (Llama via ExecuTorch, on-demand)
"""

import os
import sys
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.rule_engine import RuleEngine, RuleResult


@dataclass
class DetectionResult:
    """Final detection result combining all stages."""
    message: str
    risk_score: float
    risk_level: str          # SAFE, SUSPICIOUS, SCAM
    category: str
    matched_features: list
    rule_engine_score: float
    ml_score: Optional[float]
    explanation: Optional[str]
    safe_action: str
    stages_used: list
    latency_ms: float


# ============================================================
# Risk Score Thresholds (calibrated on validation set)
# ============================================================
THRESHOLD_SAFE = 0.30
THRESHOLD_SCAM = 0.66

# Blending weight: rule engine vs ML classifier
ALPHA = 0.4  # rule engine contribution


# ============================================================
# Safe Action Recommendations by Category
# ============================================================
SAFE_ACTIONS = {
    "fake_kyc": "Do NOT click any link. Visit your bank branch in person or use the official bank app to check KYC status.",
    "upi_fraud": "Do NOT approve any collect request you did not initiate. Open your UPI app and verify your transaction history.",
    "otp_phishing": "NEVER share your OTP with anyone. No bank or company will ever ask for your OTP. Hang up or ignore the message.",
    "prize_scam": "This is a scam. No legitimate lottery asks for fees or personal details via SMS. Delete and block.",
    "loan_extortion": "Do NOT pay through any link or number in this message. If you have a loan, contact your lender through their official app or website.",
    "bank_impersonation": "Do NOT call the number in this message. Contact your bank only through the number printed on your debit/credit card or the official website.",
    "fake_delivery": "Do NOT pay any charges through links in SMS. Track your orders only through the official app (Amazon, Flipkart, etc.).",
    "job_scam": "No legitimate job requires payment or promises unrealistic daily income. Do not share personal details or pay any fees.",
    "phishing_url": "Do NOT click the link in this message. Verify any claims through the official website or app.",
    "legitimate": "This message appears safe. No action needed.",
}


class SmartShieldPipeline:
    """
    3-Stage Hierarchical SMS Scam Detection Pipeline.
    
    Stage 1 (Rule Engine) runs on every message.
    Stage 2 (ML Classifier) runs only on ambiguous messages.
    Stage 3 (LLM Explainer) runs only on user request.
    """

    def __init__(self, patterns_path: str = None, ml_model_path: str = None,
                 llm_model_path: str = None):
        """
        Initialize the pipeline.
        
        Args:
            patterns_path: Path to scam_patterns.json
            ml_model_path: Path to TinyBERT ONNX model (optional)
            llm_model_path: Path to Llama ExecuTorch model (optional)
        """
        # Stage 1: Rule Engine (always loaded)
        self.rule_engine = RuleEngine(patterns_path)
        
        # Stage 2: ML Classifier (loaded if available)
        self.ml_model = None
        self.ml_tokenizer = None
        if ml_model_path and os.path.exists(ml_model_path):
            self._load_ml_model(ml_model_path)
        
        # Stage 3: LLM Explainer (loaded if available)
        self.llm_runner = None
        if llm_model_path and os.path.exists(llm_model_path):
            self._load_llm(llm_model_path)
        
        print("[SmartShield] Pipeline initialized")
        print(f"  Stage 1 (Rule Engine): READY")
        print(f"  Stage 2 (ML Classifier): {'READY' if self.ml_model else 'NOT LOADED (rule-engine only mode)'}")
        print(f"  Stage 3 (LLM Explainer): {'READY' if self.llm_runner else 'NOT LOADED (no explanations)'}")

    def _load_ml_model(self, model_path: str):
        """Load TinyBERT ONNX model for Stage 2."""
        try:
            import onnxruntime as ort
            self.ml_model = ort.InferenceSession(model_path)
            print(f"  [ML] ONNX model loaded from {model_path}")
        except Exception as e:
            print(f"  [ML] WARNING: Could not load ONNX model: {e}")
            self.ml_model = None

    def _load_llm(self, model_path: str):
        """Load Llama model via ExecuTorch for Stage 3."""
        try:
            # ExecuTorch loading will be implemented when model is available
            print(f"  [LLM] ExecuTorch model path: {model_path}")
            # self.llm_runner = ExecuTorchRunner(model_path)
        except Exception as e:
            print(f"  [LLM] WARNING: Could not load LLM: {e}")
            self.llm_runner = None

    def detect(self, message: str, sender: str = "",
               explain: bool = False) -> DetectionResult:
        """
        Run the full detection pipeline on a message.
        
        Args:
            message: SMS text content.
            sender: Sender ID (optional).
            explain: Whether to generate LLM explanation (Stage 3).
        
        Returns:
            DetectionResult with risk score, category, and explanation.
        """
        start_time = time.perf_counter()
        stages_used = []
        ml_score = None
        explanation = None

        # ---- STAGE 1: Rule Engine ----
        rule_result = self.rule_engine.analyze(message, sender)
        stages_used.append("rule_engine")
        rule_score = rule_result.risk_score

        # ---- STAGE 2: ML Classifier (if needed) ----
        if rule_result.needs_ml and self.ml_model:
            ml_score = self._run_ml_classifier(message)
            stages_used.append("ml_classifier")
            # Blend scores
            final_score = ALPHA * rule_score + (1 - ALPHA) * ml_score
        elif rule_result.needs_ml and not self.ml_model:
            # No ML model available — use rule score with reduced confidence
            final_score = rule_score
        else:
            # Rule engine is confident enough
            final_score = rule_score

        # ---- Determine Risk Level ----
        if final_score <= THRESHOLD_SAFE:
            risk_level = "SAFE"
        elif final_score >= THRESHOLD_SCAM:
            risk_level = "SCAM"
        else:
            risk_level = "SUSPICIOUS"

        # ---- STAGE 3: LLM Explainer (if requested) ----
        if explain and risk_level != "SAFE":
            explanation = self._generate_explanation(
                message, rule_result, final_score
            )
            if explanation:
                stages_used.append("llm_explainer")

        # ---- Safe Action ----
        safe_action = SAFE_ACTIONS.get(
            rule_result.category,
            "Exercise caution with this message. Verify through official channels."
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return DetectionResult(
            message=message[:100] + ("..." if len(message) > 100 else ""),
            risk_score=round(final_score, 3),
            risk_level=risk_level,
            category=rule_result.category,
            matched_features=rule_result.matched_features,
            rule_engine_score=rule_score,
            ml_score=ml_score,
            explanation=explanation,
            safe_action=safe_action,
            stages_used=stages_used,
            latency_ms=round(elapsed_ms, 2)
        )

    def _run_ml_classifier(self, message: str) -> float:
        """Run TinyBERT ONNX classifier on a message."""
        # TODO: Implement when ONNX model is trained
        # For now, return a placeholder
        return 0.5

    def _generate_explanation(self, message: str, rule_result: RuleResult,
                              risk_score: float) -> Optional[str]:
        """Generate LLM explanation using Llama via ExecuTorch."""
        if not self.llm_runner:
            # Fallback: template-based explanation
            return self._template_explanation(rule_result, risk_score)
        
        # TODO: ExecuTorch Llama inference
        prompt = self._build_llm_prompt(message, rule_result, risk_score)
        # response = self.llm_runner.generate(prompt, max_tokens=150)
        # return response
        return self._template_explanation(rule_result, risk_score)

    def _template_explanation(self, rule_result: RuleResult,
                               risk_score: float) -> str:
        """Fallback template-based explanation when LLM is not available."""
        category_explanations = {
            "fake_kyc": "This message pretends to be from your bank and asks you to update KYC through a suspicious link. Legitimate banks never send KYC update links via SMS.",
            "upi_fraud": "This message is trying to trick you into approving a UPI collect request disguised as a credit. In UPI, collect requests always debit money from your account.",
            "otp_phishing": "This message is asking you to share your OTP. No legitimate bank, company, or government agency will ever ask for your OTP via SMS or call.",
            "prize_scam": "This message claims you have won a prize or lottery. Legitimate lotteries never notify winners via SMS or ask for processing fees.",
            "loan_extortion": "This message is using threatening language to pressure you into paying through unofficial channels. Legitimate lenders do not threaten to contact your friends or family.",
            "bank_impersonation": "This message is impersonating your bank using a fake phone number or website. Banks never ask you to call unverified numbers from SMS.",
            "fake_delivery": "This message falsely claims your package needs a payment to release. Legitimate couriers do not demand payments via SMS links.",
            "job_scam": "This message promises unrealistic income for simple tasks. No legitimate job offers guaranteed daily income via SMS or WhatsApp.",
            "phishing_url": "This message contains a suspicious link that may steal your personal information. Do not click on it.",
        }
        
        base = category_explanations.get(
            rule_result.category,
            "This message contains suspicious patterns that may indicate a scam."
        )
        
        features_text = ""
        if rule_result.matched_features:
            features_text = f" Detected indicators: {', '.join(rule_result.matched_features[:3])}."
        
        return f"{base}{features_text} Risk score: {risk_score:.0%}."

    def _build_llm_prompt(self, message: str, rule_result: RuleResult,
                          risk_score: float) -> str:
        """Build the structured prompt for Llama ExecuTorch."""
        return f"""[SYSTEM]
You are a cybersecurity assistant running on-device. A spam detector analyzed 
an SMS and produced the following structured findings. Your job is to explain 
these findings in 2-3 simple sentences that a non-technical user can understand, 
and suggest one safe action.

Rules:
- Base your explanation ONLY on the detector findings below.
- Do NOT invent evidence or threats not found by the detector.
- Do NOT instruct the user to click any link or call any number from the message.
- Suggest a safe alternative (e.g., "Open your bank's official app directly").

[DETECTOR FINDINGS]
Risk Score: {risk_score:.2f}
Category: {rule_result.category}
Detected Features: {', '.join(rule_result.matched_features)}

[SMS]
"{message}"

[ASSISTANT]
"""


# ============================================================
# CLI Interface
# ============================================================

def print_result(result: DetectionResult, verbose: bool = False):
    """Pretty-print a detection result."""
    # Color codes
    colors = {
        "SAFE": "\033[92m",      # Green
        "SUSPICIOUS": "\033[93m", # Yellow
        "SCAM": "\033[91m",      # Red
    }
    reset = "\033[0m"
    color = colors.get(result.risk_level, "")
    
    print(f"\n{'─' * 70}")
    print(f"  Message: {result.message}")
    print(f"  {color}Risk Level: {result.risk_level} "
          f"({result.risk_score:.0%}){reset}")
    print(f"  Category: {result.category}")
    
    if result.matched_features:
        print(f"  Detected: {', '.join(result.matched_features[:4])}")
    
    print(f"  Stages: {' → '.join(result.stages_used)}")
    print(f"  Latency: {result.latency_ms:.2f} ms")
    
    if result.risk_level != "SAFE":
        print(f"\n  ⚠ Safe Action: {result.safe_action}")
    
    if result.explanation:
        print(f"\n  💡 Explanation: {result.explanation}")
    
    if verbose:
        print(f"\n  Rule Score: {result.rule_engine_score}")
        if result.ml_score is not None:
            print(f"  ML Score: {result.ml_score}")


def interactive_mode(pipeline: SmartShieldPipeline):
    """Run pipeline in interactive mode — type messages to classify."""
    print("\n" + "=" * 70)
    print("  SmartShield Interactive Mode")
    print("  Type an SMS message to analyze. Type 'quit' to exit.")
    print("  Add '!' at the end to get an explanation.")
    print("=" * 70)
    
    while True:
        try:
            msg = input("\n📱 SMS > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not msg or msg.lower() in ("quit", "exit", "q"):
            break
        
        explain = msg.endswith("!")
        if explain:
            msg = msg[:-1].strip()
        
        result = pipeline.detect(msg, explain=explain)
        print_result(result, verbose=True)


def batch_mode(pipeline: SmartShieldPipeline, input_file: str):
    """Run pipeline on a CSV file of messages."""
    import csv
    
    results = {"SAFE": 0, "SUSPICIOUS": 0, "SCAM": 0}
    correct = 0
    total = 0
    total_latency = 0
    
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = pipeline.detect(row["text"])
            results[result.risk_level] += 1
            total_latency += result.latency_ms
            total += 1
            
            # Check accuracy if labels exist
            if "label" in row:
                expected = row["label"]
                if expected == "scam" and result.risk_level in ("SCAM", "SUSPICIOUS"):
                    correct += 1
                elif expected == "legitimate" and result.risk_level == "SAFE":
                    correct += 1
    
    print(f"\n{'=' * 60}")
    print(f"  Batch Results: {input_file}")
    print(f"{'=' * 60}")
    print(f"  Total messages: {total}")
    print(f"  SAFE: {results['SAFE']}  |  SUSPICIOUS: {results['SUSPICIOUS']}  |  SCAM: {results['SCAM']}")
    if total > 0:
        print(f"  Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
        print(f"  Avg latency: {total_latency/total:.2f} ms")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="SmartShield — On-Device SMS Scam Detection"
    )
    parser.add_argument(
        "--mode", choices=["interactive", "batch", "single"],
        default="interactive", help="Run mode"
    )
    parser.add_argument("--message", "-m", type=str, help="Single message to analyze")
    parser.add_argument("--input", "-i", type=str, help="CSV file for batch mode")
    parser.add_argument("--explain", "-e", action="store_true",
                        help="Generate explanation for flagged messages")
    parser.add_argument("--patterns", type=str, help="Path to scam_patterns.json")
    parser.add_argument("--model", type=str, help="Path to ONNX model")
    parser.add_argument("--llm", type=str, help="Path to ExecuTorch Llama model")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SmartShieldPipeline(
        patterns_path=args.patterns,
        ml_model_path=args.model,
        llm_model_path=args.llm
    )
    
    if args.mode == "single" and args.message:
        result = pipeline.detect(args.message, explain=args.explain)
        print_result(result, verbose=True)
    elif args.mode == "batch" and args.input:
        batch_mode(pipeline, args.input)
    else:
        interactive_mode(pipeline)


if __name__ == "__main__":
    main()
