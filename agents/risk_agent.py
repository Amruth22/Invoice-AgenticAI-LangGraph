"""
Risk Assessment Agent for Invoice Processing
Handles fraud detection, compliance checking, and risk scoring with AI assistance
"""

import os
import json
import re
from typing import Dict, Any, List
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent
from state import (
    InvoiceProcessingState, RiskAssessment, RiskLevel,
    ValidationStatus, ProcessingStatus
)
from utils.logger import StructuredLogger

load_dotenv()


class RiskAgent(BaseAgent):
    """
    Agent responsible for risk assessment, fraud detection, and compliance checking
    Uses AI-powered analysis combined with rule-based risk factors
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("risk", config)
        self.structured_logger = StructuredLogger("risk_agent")
        
        # Initialize Gemini AI for risk assessment
        api_key = os.getenv("GEMINI_API_KEY_3")
        if not api_key:
            raise ValueError("GEMINI_API_KEY_3 not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Risk thresholds configuration
        self.risk_thresholds = config.get("risk_thresholds", {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.9
        })
        
        self.amount_thresholds = config.get("amount_thresholds", {
            "low": 1000,
            "medium": 5000,
            "high": 15000,
            "critical": 50000
        })
        
        # Fraud detection patterns
        self.fraud_patterns = config.get("fraud_patterns", [
            r"urgent.*payment",
            r"immediate.*transfer",
            r"confidential.*invoice",
            r"duplicate.*billing",
            r"rush.*order"
        ])
        
        # Compliance rules
        self.compliance_rules = config.get("compliance_rules", {
            "max_single_invoice": 100000,
            "max_monthly_customer": 500000,
            "require_approval_above": 25000,
            "blocked_customers": []
        })
    
    def _validate_preconditions(self, state: InvoiceProcessingState) -> bool:
        """Validate that we have invoice and validation data"""
        return (
            state.invoice_data is not None and
            state.validation_result is not None
        )
    
    def _validate_postconditions(self, state: InvoiceProcessingState) -> bool:
        """Validate that risk assessment was completed"""
        return (
            state.risk_assessment is not None and
            state.risk_assessment.risk_level is not None
        )
    
    async def execute(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """
        Execute risk assessment workflow
        """
        try:
            invoice_data = state.invoice_data
            validation_result = state.validation_result
            
            # Step 1: Calculate base risk score using rules
            base_risk_score = await self._calculate_base_risk_score(invoice_data, validation_result)
            
            # Step 2: Detect fraud indicators
            fraud_indicators = await self._detect_fraud_indicators(invoice_data, validation_result)
            
            # Step 3: Check compliance issues
            compliance_issues = await self._check_compliance(invoice_data, state)
            
            # Step 4: AI-powered risk assessment
            ai_risk_assessment = await self._ai_risk_assessment(invoice_data, validation_result, fraud_indicators)
            
            # Step 5: Combine all risk factors
            final_risk_score = self._combine_risk_factors(
                base_risk_score, fraud_indicators, compliance_issues, ai_risk_assessment
            )
            
            # Step 6: Determine risk level and recommendation
            risk_level = self._determine_risk_level(final_risk_score)
            recommendation = self._generate_recommendation(
                risk_level, fraud_indicators, compliance_issues, validation_result
            )
            
            # Create risk assessment
            risk_assessment = RiskAssessment(
                risk_level=risk_level,
                risk_score=final_risk_score,
                fraud_indicators=fraud_indicators,
                compliance_issues=compliance_issues,
                recommendation=recommendation["action"],
                reason=recommendation["reason"],
                requires_human_review=recommendation["requires_human_review"]
            )
            
            # Update state
            state.risk_assessment = risk_assessment
            
            # Log risk decision
            self.structured_logger.log_decision(
                agent_name=self.agent_name,
                process_id=state.process_id,
                decision=f"{risk_level.value}_{recommendation['action']}",
                reasoning=recommendation["reason"],
                confidence=final_risk_score
            )
            
            # Handle escalation
            if risk_assessment.requires_human_review or risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                state.human_review_required = True
                state.human_review_notes = f"Risk assessment: {recommendation['reason']}"
                
                if risk_level == RiskLevel.CRITICAL:
                    state.escalation_required = True
                    state.escalation_reason = f"Critical risk detected: {recommendation['reason']}"
            
            return state
            
        except Exception as e:
            state.risk_factors.append(str(e))
            self.structured_logger.log_agent_error(
                agent_name=self.agent_name,
                process_id=state.process_id,
                error=e
            )
            raise
    
    async def _calculate_base_risk_score(self, invoice_data, validation_result) -> float:
        """
        Calculate base risk score using rule-based factors
        """
        risk_factors = []
        
        # Amount-based risk
        amount_risk = min(invoice_data.total / self.amount_thresholds["critical"], 1.0)
        risk_factors.append(amount_risk * 0.3)
        
        # Validation-based risk
        if validation_result.validation_status == ValidationStatus.INVALID:
            risk_factors.append(0.4)
        elif validation_result.validation_status == ValidationStatus.REQUIRES_APPROVAL:
            risk_factors.append(0.3)
        elif validation_result.validation_status == ValidationStatus.PARTIAL_MATCH:
            risk_factors.append(0.2)
        elif not validation_result.po_found:
            risk_factors.append(0.5)
        else:
            risk_factors.append(0.0)
        
        # Discrepancy-based risk
        if validation_result.discrepancies:
            discrepancy_risk = min(len(validation_result.discrepancies) * 0.1, 0.3)
            risk_factors.append(discrepancy_risk)
        
        # Due date risk (if overdue or very urgent)
        if invoice_data.due_date:
            due_date_risk = self._calculate_due_date_risk(invoice_data.due_date)
            risk_factors.append(due_date_risk * 0.1)
        
        return sum(risk_factors)
    
    def _calculate_due_date_risk(self, due_date_str: str) -> float:
        """Calculate risk based on due date urgency"""
        try:
            # Parse due date (assuming various formats)
            due_date = self._parse_date(due_date_str)
            if not due_date:
                return 0.2  # Unknown due date is slightly risky
            
            today = datetime.now().date()
            days_diff = (due_date - today).days
            
            if days_diff < 0:  # Overdue
                return min(abs(days_diff) / 30, 1.0)  # Higher risk for longer overdue
            elif days_diff == 0:  # Due today
                return 0.3
            elif days_diff <= 3:  # Very urgent
                return 0.4
            else:
                return 0.0
                
        except Exception:
            return 0.2
    
    def _parse_date(self, date_str: str) -> datetime.date:
        """Parse date string in various formats"""
        date_formats = [
            "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
            "%b %d %Y", "%B %d %Y", "%d %b %Y"
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue
        return None
    
    async def _detect_fraud_indicators(self, invoice_data, validation_result) -> List[str]:
        """
        Detect potential fraud indicators
        """
        indicators = []
        
        # Pattern-based fraud detection
        text_to_check = f"{invoice_data.customer_name} {invoice_data.raw_text or ''}"
        for pattern in self.fraud_patterns:
            if re.search(pattern, text_to_check, re.IGNORECASE):
                indicators.append(f"Suspicious pattern detected: {pattern}")
        
        # Amount-based indicators
        if invoice_data.total > self.amount_thresholds["critical"]:
            indicators.append("Unusually high invoice amount")
        
        # Validation-based indicators
        if validation_result.discrepancies:
            overbilling = [d for d in validation_result.discrepancies if "overbilling" in d.lower()]
            if overbilling:
                indicators.append("Potential overbilling detected")
        
        # Item-based indicators
        for item in invoice_data.item_details:
            if item.rate > 10000:  # Unusually high rate
                indicators.append(f"Unusually high item rate: ${item.rate}")
            
            if item.quantity > 1000:  # Unusually high quantity
                indicators.append(f"Unusually high quantity: {item.quantity}")
        
        # Customer name anomalies
        if len(invoice_data.customer_name.split()) == 1:  # Single word customer name
            indicators.append("Suspicious customer name format")
        
        # Duplicate detection (simplified)
        if hasattr(self, '_processed_invoices'):
            for processed in self._processed_invoices:
                if (processed.get('customer_name') == invoice_data.customer_name and
                    abs(processed.get('total', 0) - invoice_data.total) < 0.01):
                    indicators.append("Potential duplicate invoice")
        
        return indicators
    
    async def _check_compliance(self, invoice_data, state: InvoiceProcessingState) -> List[str]:
        """
        Check compliance against business rules
        """
        issues = []
        
        # Amount limits
        if invoice_data.total > self.compliance_rules["max_single_invoice"]:
            issues.append(f"Exceeds maximum single invoice limit: ${self.compliance_rules['max_single_invoice']}")
        
        # Approval requirements
        if invoice_data.total > self.compliance_rules["require_approval_above"]:
            issues.append(f"Requires approval for amounts above ${self.compliance_rules['require_approval_above']}")
        
        # Blocked customers
        if invoice_data.customer_name in self.compliance_rules["blocked_customers"]:
            issues.append("Customer is on blocked list")
        
        # Missing required fields
        required_fields = ["invoice_number", "customer_name", "total"]
        for field in required_fields:
            if not getattr(invoice_data, field, None):
                issues.append(f"Missing required field: {field}")
        
        # Business hours check (if urgent payment)
        current_hour = datetime.now().hour
        if current_hour < 9 or current_hour > 17:  # Outside business hours
            if state.priority_level > 3:
                issues.append("High priority payment requested outside business hours")
        
        return issues
    
    async def _ai_risk_assessment(self, invoice_data, validation_result, fraud_indicators: List[str]) -> Dict[str, Any]:
        """
        Use AI to assess risk based on context and patterns
        """
        # Prepare context for AI
        context = {
            "invoice_number": invoice_data.invoice_number,
            "customer_name": invoice_data.customer_name,
            "amount": invoice_data.total,
            "validation_status": validation_result.validation_status.value,
            "discrepancies": validation_result.discrepancies,
            "fraud_indicators": fraud_indicators
        }
        
        prompt = f"""
You are a financial risk analyst. Assess the risk level for this invoice payment based on the provided context.

Context:
{json.dumps(context, indent=2)}

Consider:
1. Validation discrepancies and their severity
2. Fraud indicators and suspicious patterns
3. Amount relative to typical business transactions
4. Customer credibility factors
5. Overall payment risk

Respond ONLY in valid JSON format:
{{
  "ai_risk_score": 0.0-1.0,
  "risk_factors": ["factor1", "factor2"],
  "recommendation": "approve" | "investigate" | "escalate" | "hold",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of the assessment"
}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            content = self._clean_json_response(response.text)
            ai_assessment = json.loads(content)
            
            # Validate AI response
            if not all(key in ai_assessment for key in ["ai_risk_score", "recommendation", "reasoning"]):
                raise ValueError("Incomplete AI response")
            
            return ai_assessment
            
        except Exception as e:
            self.logger.warning(f"AI risk assessment failed: {e}")
            # Fallback to rule-based assessment
            return {
                "ai_risk_score": 0.5,
                "risk_factors": ["AI assessment unavailable"],
                "recommendation": "investigate",
                "confidence": 0.3,
                "reasoning": f"AI assessment failed: {str(e)}"
            }
    
    def _clean_json_response(self, text: str) -> str:
        """Clean AI response to extract valid JSON"""
        text = text.strip()
        text = re.sub(r"```json|```", "", text)
        text = re.sub(r"//.*", "", text)
        text = text.replace("'", '"')
        return text
    
    def _combine_risk_factors(self, base_score: float, fraud_indicators: List[str], 
                            compliance_issues: List[str], ai_assessment: Dict[str, Any]) -> float:
        """
        Combine all risk factors into final risk score
        """
        # Base risk (40% weight)
        final_score = base_score * 0.4
        
        # Fraud indicators (25% weight)
        fraud_score = min(len(fraud_indicators) * 0.15, 1.0)
        final_score += fraud_score * 0.25
        
        # Compliance issues (20% weight)
        compliance_score = min(len(compliance_issues) * 0.2, 1.0)
        final_score += compliance_score * 0.2
        
        # AI assessment (15% weight)
        ai_score = ai_assessment.get("ai_risk_score", 0.5)
        final_score += ai_score * 0.15
        
        return min(final_score, 1.0)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on score"""
        if risk_score >= self.risk_thresholds["critical"]:
            return RiskLevel.CRITICAL
        elif risk_score >= self.risk_thresholds["high"]:
            return RiskLevel.HIGH
        elif risk_score >= self.risk_thresholds["medium"]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_recommendation(self, risk_level: RiskLevel, fraud_indicators: List[str],
                               compliance_issues: List[str], validation_result) -> Dict[str, Any]:
        """
        Generate recommendation based on risk assessment
        """
        if risk_level == RiskLevel.CRITICAL:
            return {
                "action": "hold",
                "reason": f"Critical risk detected: {'; '.join(fraud_indicators + compliance_issues)}",
                "requires_human_review": True
            }
        
        elif risk_level == RiskLevel.HIGH:
            return {
                "action": "escalate",
                "reason": f"High risk factors: {'; '.join(fraud_indicators + compliance_issues)}",
                "requires_human_review": True
            }
        
        elif risk_level == RiskLevel.MEDIUM:
            if validation_result.validation_status == ValidationStatus.VALID:
                return {
                    "action": "approve",
                    "reason": "Medium risk but validation passed",
                    "requires_human_review": False
                }
            else:
                return {
                    "action": "investigate",
                    "reason": "Medium risk with validation issues",
                    "requires_human_review": True
                }
        
        else:  # LOW risk
            return {
                "action": "approve",
                "reason": "Low risk assessment",
                "requires_human_review": False
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for risk agent"""
        health_status = await super().health_check()
        
        try:
            # Test AI connection
            test_response = self.model.generate_content("Test risk assessment connection")
            ai_status = "healthy" if test_response else "unhealthy"
        except Exception as e:
            ai_status = f"unhealthy: {str(e)}"
        
        health_status.update({
            "ai_model_status": ai_status,
            "risk_thresholds": self.risk_thresholds,
            "compliance_rules_loaded": bool(self.compliance_rules)
        })
        
        return health_status