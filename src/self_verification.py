"""
Self-Verification Module for Clinical Decision Support.
Cross-verifies primary deep learning predictions against radiological nodule characteristics,
detecting potential false positives, false negatives, and uncertainty anomalies.
"""

from typing import Dict, List, Any


class SelfVerificationEngine:
    """
    Validates model predictions using clinical radiological characteristics:
    [0: Spiculation, 1: Lobulation, 2: Calcification, 3: Subtlety, 4: Margin, 5: Sphericity]
    """
    def __init__(self, uncertainty_threshold: float = 0.08, malignant_threshold: float = 0.50):
        self.uncertainty_threshold = uncertainty_threshold
        self.malignant_threshold = malignant_threshold

    def verify_prediction(
        self,
        calibrated_prob: float,
        epistemic_uncertainty: float,
        traits: List[float]
    ) -> Dict[str, Any]:
        """
        Executes adaptive self-verification rules.
        """
        spiculation, lobulation, calcification, subtlety, margin, sphericity = traits
        
        primary_is_cancerous = calibrated_prob >= self.malignant_threshold
        
        # Assess radiological characteristic risk score (0 to 1)
        # High spiculation, high lobulation, low margin (poorly defined), absent calcification (6) increase malignant risk
        trait_risk_components = {
            "Spiculation": (spiculation - 1.0) / 4.0,           # Higher = Malignant
            "Lobulation": (lobulation - 1.0) / 4.0,             # Higher = Malignant
            "Poor Margin": (5.0 - margin) / 4.0,                # Lower margin = Malignant
            "Irregularity": (5.0 - sphericity) / 4.0,          # Lower sphericity = Malignant
            "Calcification Pattern": (calcification - 1.0) / 5.0 # High (6=absent) = Malignant
        }
        
        trait_risk_score = float(sum(trait_risk_components.values()) / len(trait_risk_components))
        
        conflicts = []
        status = "VERIFIED_MATCH"
        
        # Rule 1: Epistemic Uncertainty Check
        if epistemic_uncertainty > self.uncertainty_threshold:
            status = "UNCERTAIN_HIGH_VARIANCE"
            conflicts.append(
                f"High epistemic model uncertainty ({epistemic_uncertainty:.4f} > {self.uncertainty_threshold:.4f}). Monte Carlo predictions varied."
            )
            
        # Rule 2: False Positive Risk (Primary Cancerous, but Benign Traits)
        if primary_is_cancerous and trait_risk_score < 0.35:
            if status != "UNCERTAIN_HIGH_VARIANCE":
                status = "WARNING_POSSIBLE_FALSE_POSITIVE"
            conflicts.append(
                f"Conflict: Primary model predicted Cancerous ({calibrated_prob*100:.1f}%), "
                f"but radiological traits indicate benign characteristics (Trait Risk Score: {trait_risk_score:.2f}). "
                f"Margin is smooth ({margin:.1f}/5) and spiculation is low ({spiculation:.1f}/5)."
            )
            
        # Rule 3: False Negative Risk (Primary Benign, but Malignant Traits)
        if not primary_is_cancerous and trait_risk_score > 0.65:
            if status != "UNCERTAIN_HIGH_VARIANCE":
                status = "WARNING_POSSIBLE_FALSE_NEGATIVE"
            conflicts.append(
                f"Conflict: Primary model predicted Non-Cancerous ({calibrated_prob*100:.1f}%), "
                f"however radiological traits show high malignancy indicators (Trait Risk Score: {trait_risk_score:.2f}). "
                f"Spiculation is elevated ({spiculation:.1f}/5) and margin is invasive ({margin:.1f}/5)."
            )

        # Final Clinical Decision Synthesis
        if status == "VERIFIED_MATCH":
            if primary_is_cancerous:
                final_diagnosis = "High Risk - Malignant Pulmonary Nodule"
                action = "Urgent consultation with pulmonologist/oncologist. Recommend contrast-enhanced CT or PET-CT scan."
            else:
                final_diagnosis = "Low Risk - Benign Pulmonary Nodule"
                action = "Routine follow-up low-dose CT scan recommended in 6 to 12 months."
        elif status == "WARNING_POSSIBLE_FALSE_POSITIVE":
            final_diagnosis = "Inconclusive (Flagged Potential False Positive)"
            action = "Radiological verification conflict detected. Specialist manual review required before invasive procedures."
        elif status == "WARNING_POSSIBLE_FALSE_NEGATIVE":
            final_diagnosis = "High Risk (Flagged Potential False Negative)"
            action = "Primary classification benign, but radiological traits show high-risk spiculation. Short-interval 3-month CT scan or biopsy recommended."
        else:
            final_diagnosis = "Inconclusive - High Prediction Uncertainty"
            action = "Model confidence is uncalibrated/uncertain. Clinical evaluation by senior thoracic radiologist strongly advised."

        return {
            "final_diagnosis": final_diagnosis,
            "status": status,
            "is_verified": status == "VERIFIED_MATCH",
            "primary_probability": calibrated_prob,
            "trait_risk_score": trait_risk_score,
            "epistemic_uncertainty": epistemic_uncertainty,
            "conflicts": conflicts,
            "recommended_action": action,
            "trait_breakdown": {
                "Spiculation (1-5)": f"{spiculation:.2f}",
                "Lobulation (1-5)": f"{lobulation:.2f}",
                "Calcification (1-6)": f"{calcification:.2f}",
                "Subtlety (1-5)": f"{subtlety:.2f}",
                "Margin (1-5)": f"{margin:.2f}",
                "Sphericity (1-5)": f"{sphericity:.2f}"
            }
        }
