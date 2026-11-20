"""
SmartShield Rule Engine — Stage 1 Detection
India-specific SMS scam pattern matching with zero-latency regex rules.

This module catches ~65-70% of obvious scams in <1ms without any ML model.
Ambiguous messages are passed to Stage 2 (TinyBERT classifier).
"""

import re
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class RuleResult:
    """Result from the rule engine analysis."""
    risk_score: float                    # 0.0 (safe) to 1.0 (scam)
    category: str                       # Threat category or "legitimate"
    confidence: str                     # "high", "medium", "low"
    matched_features: List[str]         # List of features that fired
    needs_ml: bool                      # Whether to pass to Stage 2
    explanation_data: dict = field(default_factory=dict)  # Structured data for LLM


class RuleEngine:
    """
    Stage 1: Deterministic rule-based SMS scam detector.
    
    Uses regex pattern matching, keyword detection, and structural
    feature extraction to classify obvious scams and legitimate messages.
    Ambiguous messages are forwarded to the ML classifier (Stage 2).
    """

    def __init__(self, patterns_path: str = None):
        if patterns_path is None:
            patterns_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "rules", "scam_patterns.json"
            )
        
        with open(patterns_path, "r", encoding="utf-8") as f:
            self.patterns = json.load(f)
        
        # Pre-compile all regex patterns for performance
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for fast matching."""
        self.compiled = {}
        
        pattern_groups = [
            "otp_phishing_patterns", "kyc_scam_patterns",
            "upi_fraud_patterns", "prize_lottery_patterns",
            "loan_extortion_patterns", "job_scam_patterns",
            "suspicious_url_patterns", "bank_impersonation_patterns",
            "fake_delivery_patterns", "legitimate_indicators"
        ]
        
        for group in pattern_groups:
            if group in self.patterns:
                self.compiled[group] = [
                    re.compile(p, re.IGNORECASE) for p in self.patterns[group]
                ]
        
        # Compile keyword lists into single alternation patterns
        for key in ["urgency_keywords", "threat_keywords"]:
            if key in self.patterns:
                all_words = []
                for lang_words in self.patterns[key].values():
                    all_words.extend(lang_words)
                escaped = [re.escape(w) for w in all_words]
                self.compiled[key] = re.compile(
                    r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE
                )

    def analyze(self, message: str, sender: str = "") -> RuleResult:
        """
        Analyze an SMS message for scam indicators.
        
        Args:
            message: The SMS text content.
            sender: The sender ID (e.g., "VM-SBIINB", phone number).
        
        Returns:
            RuleResult with risk score, category, and matched features.
        """
        start_time = time.perf_counter_ns()
        
        features = []
        category_scores = {}
        text_lower = message.lower()
        
        # ---- Feature Extraction ----
        
        # 1. Structural features
        structural = self._extract_structural_features(message)
        
        # 2. OTP phishing
        otp_hits = self._match_group("otp_phishing_patterns", text_lower)
        if otp_hits:
            features.append(f"asks_for_otp: {otp_hits[0]}")
            category_scores["otp_phishing"] = 0.9
        
        # 3. KYC scam
        kyc_hits = self._match_group("kyc_scam_patterns", text_lower)
        if kyc_hits:
            features.append(f"fake_kyc_pattern: {kyc_hits[0]}")
            category_scores["fake_kyc"] = 0.85
        
        # 4. UPI fraud
        upi_hits = self._match_group("upi_fraud_patterns", text_lower)
        if upi_hits:
            features.append(f"upi_collect_fraud: {upi_hits[0]}")
            category_scores["upi_fraud"] = 0.9
        
        # 5. Prize/lottery
        prize_hits = self._match_group("prize_lottery_patterns", text_lower)
        if prize_hits:
            features.append(f"prize_lottery_scam: {prize_hits[0]}")
            category_scores["prize_scam"] = 0.95
        
        # 6. Loan extortion
        loan_hits = self._match_group("loan_extortion_patterns", text_lower)
        if loan_hits:
            features.append(f"loan_extortion: {loan_hits[0]}")
            category_scores["loan_extortion"] = 0.9
        
        # 7. Job scam
        job_hits = self._match_group("job_scam_patterns", text_lower)
        if job_hits:
            features.append(f"job_income_scam: {job_hits[0]}")
            category_scores["job_scam"] = 0.85
        
        # 8. Bank impersonation
        bank_hits = self._match_group("bank_impersonation_patterns", text_lower)
        if bank_hits:
            features.append(f"bank_impersonation: {bank_hits[0]}")
            category_scores["bank_impersonation"] = 0.8
        
        # 9. Fake delivery
        delivery_hits = self._match_group("fake_delivery_patterns", text_lower)
        if delivery_hits:
            features.append(f"fake_delivery: {delivery_hits[0]}")
            category_scores["fake_delivery"] = 0.85
        
        # 10. Suspicious URLs
        url_hits = self._match_group("suspicious_url_patterns", text_lower)
        if url_hits:
            features.append(f"suspicious_url: {url_hits[0]}")
            category_scores.setdefault("phishing_url", 0.0)
            category_scores["phishing_url"] = max(
                category_scores["phishing_url"], 0.75
            )
        
        # 11. Urgency keywords
        if "urgency_keywords" in self.compiled:
            urgency_matches = self.compiled["urgency_keywords"].findall(text_lower)
            if urgency_matches:
                features.append(f"urgency_language: {', '.join(urgency_matches[:3])}")
                for cat in category_scores:
                    category_scores[cat] = min(1.0, category_scores[cat] + 0.05)
        
        # 12. Threat language
        if "threat_keywords" in self.compiled:
            threat_matches = self.compiled["threat_keywords"].findall(text_lower)
            if threat_matches:
                features.append(f"threat_language: {', '.join(threat_matches[:3])}")
                for cat in category_scores:
                    category_scores[cat] = min(1.0, category_scores[cat] + 0.05)
        
        # 13. Structural risk indicators
        if structural["special_char_ratio"] > 0.15:
            features.append(f"high_special_char_ratio: {structural['special_char_ratio']:.2f}")
        if structural["uppercase_ratio"] > 0.4 and len(message) > 20:
            features.append(f"excessive_uppercase: {structural['uppercase_ratio']:.2f}")
        if structural["link_count"] > 0:
            features.append(f"contains_links: {structural['link_count']}")
        if structural["phone_numbers"]:
            features.append(f"contains_phone_number: {len(structural['phone_numbers'])} found")
        
        # 14. Legitimate message indicators (reduce score)
        legit_hits = self._match_group("legitimate_indicators", text_lower)
        legit_bonus = len(legit_hits) * 0.15
        
        # ---- Score Calculation ----
        
        if category_scores:
            top_category = max(category_scores, key=category_scores.get)
            raw_score = category_scores[top_category]
        else:
            top_category = "legitimate"
            raw_score = 0.1 + (structural["link_count"] * 0.1)
        
        # Apply legitimate indicator discount
        final_score = max(0.0, min(1.0, raw_score - legit_bonus))
        
        # ---- Decision Logic ----
        
        if final_score >= 0.85 and len(features) >= 3:
            confidence = "high"
            needs_ml = False
        elif len(legit_hits) >= 2 and len(features) == 0:
            confidence = "high"
            needs_ml = False
            top_category = "legitimate"
        else:
            confidence = "medium" if len(features) >= 1 else "low"
            needs_ml = True
        
        elapsed_us = (time.perf_counter_ns() - start_time) / 1000
        
        return RuleResult(
            risk_score=round(final_score, 3),
            category=top_category,
            confidence=confidence,
            matched_features=features,
            needs_ml=needs_ml,
            explanation_data={
                "category": top_category,
                "features": features,
                "structural": structural,
                "legitimate_indicators": legit_hits,
                "rule_engine_latency_us": round(elapsed_us, 1)
            }
        )
    
    def _match_group(self, group_name: str, text: str) -> List[str]:
        """Match text against a compiled pattern group, return matched strings."""
        matches = []
        if group_name in self.compiled:
            for pattern in self.compiled[group_name]:
                match = pattern.search(text)
                if match:
                    matches.append(match.group())
        return matches
    
    def _extract_structural_features(self, message: str) -> dict:
        """Extract structural/statistical features from the message."""
        length = max(len(message), 1)
        
        # Character ratios
        special_chars = sum(1 for c in message if not c.isalnum() and not c.isspace())
        uppercase_chars = sum(1 for c in message if c.isupper())
        
        # URLs
        urls = re.findall(
            r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.(?:com|in|xyz|tk|ml|org|net|co\.in)/[^\s]*',
            message
        )
        
        # Phone numbers (Indian format)
        phone_numbers = re.findall(
            r'(?:\+91[\s-]?)?[6-9]\d{9}', message
        )
        
        # Money amounts (Indian Rupee)
        money_mentions = re.findall(
            r'(?:rs\.?|inr|₹)\s*[\d,]+(?:\.\d{2})?', message, re.IGNORECASE
        )
        
        return {
            "length": length,
            "special_char_ratio": round(special_chars / length, 3),
            "uppercase_ratio": round(uppercase_chars / length, 3),
            "link_count": len(urls),
            "urls": urls,
            "phone_numbers": phone_numbers,
            "money_mentions": money_mentions,
            "word_count": len(message.split())
        }


# ============================================================
# Quick self-test
# ============================================================
if __name__ == "__main__":
    engine = RuleEngine()
    
    test_messages = [
        # --- SCAM messages ---
        (
            "Dear SBI customer, your KYC has expired. Update immediately "
            "or your account will be blocked. Click: http://sbi-kyc-update.xyz/verify",
            "SCAM"
        ),
        (
            "Congratulations! Your Jio number has won Rs.25,00,000 in lucky draw! "
            "Claim your prize now: bit.ly/jio-prize-claim",
            "SCAM"
        ),
        (
            "Rs.15,000 credited to your account. Approve the collect request "
            "to receive the amount in your UPI wallet.",
            "SCAM"
        ),
        (
            "URGENT: Pay Rs.5000 within 2 hours or your contacts list "
            "will be notified about your loan default. Call 9876543210",
            "SCAM"
        ),
        (
            "Share the OTP sent to your mobile number for account verification. "
            "Your HDFC account will be suspended if not verified.",
            "SCAM"
        ),
        (
            "Aapka SBI account block ho gaya hai. Turant call karein "
            "9876543210 ya link pe click karein: http://sbi-verify.xyz",
            "SCAM"
        ),
        (
            "Earn Rs.5000-10000 daily from home! Simple typing job. "
            "WhatsApp us: 9876543210. Limited seats!",
            "SCAM"
        ),
        (
            "Your Amazon parcel is held at customs. Pay Rs.49 delivery "
            "charges to release: http://amzn-delivery.in/pay",
            "SCAM"
        ),
        
        # --- LEGITIMATE messages ---
        (
            "INR 2,500.00 debited from A/c XX4532 on 15-Nov-2026. "
            "UPI Ref No: 433298765432. If not done by you, call 1800-111-109. -SBI",
            "LEGIT"
        ),
        (
            "Your OTP for SBI Net Banking login is 847263. Valid for 5 minutes. "
            "Do not share this OTP with anyone. -SBI",
            "LEGIT"
        ),
        (
            "Rs.500.00 received from XXXXX@upi in your A/c XX7891. "
            "UPI Ref No: 433298765000. -HDFC Bank",
            "LEGIT"
        ),
        (
            "Your Flipkart order #OD432876 has been shipped via Ekart. "
            "Track: https://www.flipkart.com/track/OD432876",
            "LEGIT"
        ),
        (
            "Reminder: As per RBI guidelines, please complete your KYC "
            "at the nearest SBI branch before 31-Dec-2026. -SBI",
            "LEGIT"
        ),
        (
            "Your Swiggy order #SW9988 is out for delivery. "
            "Track live on the Swiggy app.",
            "LEGIT"
        ),
    ]
    
    print("=" * 80)
    print("SmartShield Rule Engine — Self-Test")
    print("=" * 80)
    
    correct = 0
    total = len(test_messages)
    
    for msg, expected in test_messages:
        result = engine.analyze(msg)
        
        if expected == "SCAM":
            predicted = "SCAM" if result.risk_score >= 0.5 else "LEGIT"
        else:
            predicted = "LEGIT" if result.risk_score < 0.5 else "SCAM"
        
        is_correct = predicted == expected
        correct += int(is_correct)
        
        status = "✓" if is_correct else "✗"
        print(f"\n{status} [{expected}] Score: {result.risk_score:.3f} | "
              f"Category: {result.category} | Confidence: {result.confidence}")
        print(f"  Message: {msg[:80]}...")
        if result.matched_features:
            print(f"  Features: {', '.join(result.matched_features[:3])}")
        print(f"  Needs ML: {result.needs_ml} | "
              f"Latency: {result.explanation_data['rule_engine_latency_us']:.0f} µs")
    
    print(f"\n{'=' * 80}")
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")
    print(f"{'=' * 80}")
