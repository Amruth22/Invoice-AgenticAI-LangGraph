"""
Payment Agent for Invoice Processing
Handles payment decisions, processing, and transaction management
"""

import os
import json
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv

from agents.base_agent import BaseAgent
from state import (
    InvoiceProcessingState, PaymentDecision, PaymentStatus,
    RiskLevel, ValidationStatus, ProcessingStatus
)
from utils.logger import StructuredLogger

load_dotenv()


class PaymentAgent(BaseAgent):
    """
    Agent responsible for payment processing decisions and execution
    Integrates with payment systems and provides intelligent payment routing
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("payment", config)
        self.structured_logger = StructuredLogger("payment_agent")
        
        # Initialize Gemini AI for payment justification
        api_key = os.getenv("GEMINI_API_KEY_2")
        if not api_key:
            raise ValueError("GEMINI_API_KEY_2 not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Payment configuration
        self.payment_api_url = config.get("payment_api_url", "http://localhost:8000/initiate_payment")
        self.auto_payment_threshold = config.get("auto_payment_threshold", 5000)
        self.manual_approval_threshold = config.get("manual_approval_threshold", 25000)
        self.payment_timeout = config.get("payment_timeout", 30)
        
        # Payment methods configuration
        self.payment_methods = config.get("payment_methods", {
            "ach": {"limit": 100000, "processing_days": 3},
            "wire": {"limit": 1000000, "processing_days": 1},
            "check": {"limit": 50000, "processing_days": 5},
            "card": {"limit": 10000, "processing_days": 1}
        })
        
        # Retry configuration
        self.max_payment_retries = config.get("max_payment_retries", 3)
        self.retry_delay_seconds = config.get("retry_delay_seconds", 5)
    
    def _validate_preconditions(self, state: InvoiceProcessingState) -> bool:
        """Validate that we have all required data for payment processing"""
        return (
            state.invoice_data is not None and
            state.validation_result is not None and
            state.risk_assessment is not None and
            state.invoice_data.total > 0
        )
    
    def _validate_postconditions(self, state: InvoiceProcessingState) -> bool:
        """Validate that payment decision was made"""
        return (
            state.payment_decision is not None and
            state.payment_decision.payment_status is not None
        )
    
    async def execute(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """
        Execute payment processing workflow
        """
        try:
            invoice_data = state.invoice_data
            validation_result = state.validation_result
            risk_assessment = state.risk_assessment
            
            # Step 1: Make payment decision
            payment_decision = await self._make_payment_decision(
                invoice_data, validation_result, risk_assessment, state
            )
            
            # Step 2: Execute payment if approved
            if payment_decision.payment_status == PaymentStatus.APPROVED:
                payment_result = await self._execute_payment(invoice_data, payment_decision)
                payment_decision = self._update_payment_decision(payment_decision, payment_result)
            
            # Step 3: Generate payment justification
            justification = await self._generate_payment_justification(
                invoice_data, payment_decision, validation_result, risk_assessment
            )
            
            # Update state
            state.payment_decision = payment_decision
            
            # Log payment decision
            self.structured_logger.log_decision(
                agent_name=self.agent_name,
                process_id=state.process_id,
                decision=payment_decision.payment_status.value,
                reasoning=justification,
                confidence=1.0 - risk_assessment.risk_score
            )
            
            # Handle escalation for rejected/failed payments
            if payment_decision.payment_status in [PaymentStatus.REJECTED, PaymentStatus.FAILED]:
                if payment_decision.rejection_reason and "critical" in payment_decision.rejection_reason.lower():
                    state.escalation_required = True
                    state.escalation_reason = f"Payment rejected: {payment_decision.rejection_reason}"
            
            return state
            
        except Exception as e:
            state.payment_errors.append(str(e))
            self.structured_logger.log_agent_error(
                agent_name=self.agent_name,
                process_id=state.process_id,
                error=e
            )
            raise
    
    async def _make_payment_decision(self, invoice_data, validation_result, 
                                   risk_assessment, state: InvoiceProcessingState) -> PaymentDecision:
        """
        Make intelligent payment decision based on all available data
        """
        payment_decision = PaymentDecision()
        
        # Decision logic based on risk and validation
        if risk_assessment.risk_level == RiskLevel.CRITICAL:
            payment_decision.payment_status = PaymentStatus.REJECTED
            payment_decision.rejection_reason = f"Critical risk: {risk_assessment.reason}"
            
        elif risk_assessment.risk_level == RiskLevel.HIGH:
            if validation_result.validation_status == ValidationStatus.VALID:
                payment_decision.payment_status = PaymentStatus.REQUIRES_ESCALATION
                payment_decision.rejection_reason = "High risk but valid - requires approval"
            else:
                payment_decision.payment_status = PaymentStatus.REJECTED
                payment_decision.rejection_reason = f"High risk with validation issues: {validation_result.validation_result}"
        
        elif validation_result.validation_status == ValidationStatus.INVALID:
            payment_decision.payment_status = PaymentStatus.REJECTED
            payment_decision.rejection_reason = f"Validation failed: {validation_result.validation_result}"
            
        elif validation_result.validation_status == ValidationStatus.MISSING_PO:
            if invoice_data.total > self.manual_approval_threshold:
                payment_decision.payment_status = PaymentStatus.REQUIRES_ESCALATION
                payment_decision.rejection_reason = "Missing PO for high-value invoice"
            else:
                payment_decision.payment_status = PaymentStatus.REJECTED
                payment_decision.rejection_reason = "No matching purchase order found"
        
        elif invoice_data.total > self.manual_approval_threshold:
            payment_decision.payment_status = PaymentStatus.REQUIRES_ESCALATION
            payment_decision.rejection_reason = f"Amount exceeds manual approval threshold: ${self.manual_approval_threshold}"
        
        else:
            # Approve payment
            payment_decision.payment_status = PaymentStatus.APPROVED
            payment_decision.approved_amount = invoice_data.total
            
            # Select payment method
            payment_decision.payment_method = self._select_payment_method(invoice_data.total)
            
            # Schedule payment date
            payment_decision.scheduled_date = self._calculate_payment_date(
                invoice_data.due_date, payment_decision.payment_method
            )
        
        # Add to approval chain
        payment_decision.approval_chain.append(f"payment_agent_{datetime.now().isoformat()}")
        
        return payment_decision
    
    def _select_payment_method(self, amount: float) -> str:
        """
        Select optimal payment method based on amount and constraints
        """
        # Sort payment methods by processing speed and limits
        suitable_methods = [
            (method, config) for method, config in self.payment_methods.items()
            if config["limit"] >= amount
        ]
        
        if not suitable_methods:
            return "wire"  # Default for very high amounts
        
        # Prefer faster methods for smaller amounts
        if amount <= 1000:
            return "card"
        elif amount <= 10000:
            return "ach"
        else:
            return "wire"
    
    def _calculate_payment_date(self, due_date_str: Optional[str], payment_method: str) -> datetime:
        """
        Calculate optimal payment date based on due date and processing time
        """
        try:
            if due_date_str:
                due_date = self._parse_date(due_date_str)
                if due_date:
                    processing_days = self.payment_methods.get(payment_method, {}).get("processing_days", 1)
                    optimal_date = due_date - timedelta(days=processing_days)
                    
                    # Don't schedule in the past
                    today = datetime.now().date()
                    if optimal_date < today:
                        return datetime.combine(today, datetime.min.time())
                    
                    return datetime.combine(optimal_date, datetime.min.time())
        except Exception:
            pass
        
        # Default to immediate processing
        return datetime.now()
    
    def _parse_date(self, date_str: str) -> Optional[datetime.date]:
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
    
    async def _execute_payment(self, invoice_data, payment_decision: PaymentDecision) -> Dict[str, Any]:
        """
        Execute payment through payment API with retry logic
        """
        payload = {
            "order_id": invoice_data.order_id,
            "customer_name": invoice_data.customer_name,
            "amount": payment_decision.approved_amount,
            "due_date": invoice_data.due_date or datetime.now().isoformat(),
            "payment_method": payment_decision.payment_method,
            "invoice_number": invoice_data.invoice_number
        }
        
        for attempt in range(self.max_payment_retries):
            try:
                self.logger.info(f"Payment attempt {attempt + 1} for invoice {invoice_data.invoice_number}")
                
                response = requests.post(
                    self.payment_api_url,
                    json=payload,
                    timeout=self.payment_timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.logger.info(f"Payment successful: {result.get('transaction_id')}")
                    return {
                        "status": "success",
                        "transaction_id": result.get("transaction_id"),
                        "timestamp": result.get("timestamp"),
                        "message": result.get("message", "Payment processed successfully"),
                        "attempt": attempt + 1
                    }
                else:
                    error_msg = f"Payment API error: HTTP {response.status_code}"
                    self.logger.warning(f"{error_msg} on attempt {attempt + 1}")
                    
                    if attempt == self.max_payment_retries - 1:
                        return {
                            "status": "failed",
                            "error": error_msg,
                            "attempt": attempt + 1
                        }
            
            except requests.exceptions.Timeout:
                error_msg = "Payment API timeout"
                self.logger.warning(f"{error_msg} on attempt {attempt + 1}")
                
                if attempt == self.max_payment_retries - 1:
                    return {
                        "status": "failed",
                        "error": error_msg,
                        "attempt": attempt + 1
                    }
            
            except Exception as e:
                error_msg = f"Payment execution error: {str(e)}"
                self.logger.error(f"{error_msg} on attempt {attempt + 1}")
                
                if attempt == self.max_payment_retries - 1:
                    return {
                        "status": "failed",
                        "error": error_msg,
                        "attempt": attempt + 1
                    }
            
            # Wait before retry
            if attempt < self.max_payment_retries - 1:
                await self._async_sleep(self.retry_delay_seconds)
        
        return {
            "status": "failed",
            "error": "Max retries exceeded",
            "attempt": self.max_payment_retries
        }
    
    async def _async_sleep(self, seconds: int):
        """Async sleep for retry delays"""
        import asyncio
        await asyncio.sleep(seconds)
    
    def _update_payment_decision(self, payment_decision: PaymentDecision, 
                               payment_result: Dict[str, Any]) -> PaymentDecision:
        """
        Update payment decision based on execution result
        """
        if payment_result["status"] == "success":
            payment_decision.payment_status = PaymentStatus.PROCESSED
            payment_decision.transaction_id = payment_result.get("transaction_id")
        else:
            payment_decision.payment_status = PaymentStatus.FAILED
            payment_decision.rejection_reason = payment_result.get("error", "Payment execution failed")
        
        return payment_decision
    
    async def _generate_payment_justification(self, invoice_data, payment_decision: PaymentDecision,
                                            validation_result, risk_assessment) -> str:
        """
        Generate AI-powered payment justification for audit trail
        """
        context = {
            "invoice_number": invoice_data.invoice_number,
            "customer_name": invoice_data.customer_name,
            "amount": invoice_data.total,
            "payment_status": payment_decision.payment_status.value,
            "validation_status": validation_result.validation_status.value,
            "risk_level": risk_assessment.risk_level.value,
            "transaction_id": payment_decision.transaction_id
        }
        
        prompt = f"""
You are a financial controller generating a payment justification for audit records.

Context:
{json.dumps(context, indent=2)}

Generate a concise, professional justification statement explaining the payment decision.

For approved/processed payments, focus on validation success and risk assessment.
For rejected payments, explain the specific reasons and compliance requirements.
For escalated payments, note the approval requirements.

Respond with a single professional sentence suitable for audit logs.
Do not use markdown or extra formatting.
"""
        
        try:
            response = self.model.generate_content(prompt)
            justification = response.text.strip()
            
            # Clean up the response
            justification = justification.replace("**", "").replace("*", "")
            if justification.startswith('"') and justification.endswith('"'):
                justification = justification[1:-1]
            
            return justification
            
        except Exception as e:
            self.logger.warning(f"AI justification failed: {e}")
            
            # Fallback to template-based justification
            if payment_decision.payment_status == PaymentStatus.PROCESSED:
                return f"Payment of ${invoice_data.total:.2f} processed for {invoice_data.customer_name}. Invoice {invoice_data.invoice_number} validated with {risk_assessment.risk_level.value} risk level."
            elif payment_decision.payment_status == PaymentStatus.REJECTED:
                return f"Payment of ${invoice_data.total:.2f} rejected for {invoice_data.customer_name}. Reason: {payment_decision.rejection_reason}"
            else:
                return f"Payment of ${invoice_data.total:.2f} for {invoice_data.customer_name} requires escalation. Status: {payment_decision.payment_status.value}"
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for payment agent"""
        health_status = await super().health_check()
        
        # Test payment API connectivity
        try:
            test_payload = {
                "order_id": "HEALTH_CHECK",
                "customer_name": "Test Customer",
                "amount": 1.00,
                "due_date": datetime.now().isoformat()
            }
            
            response = requests.post(
                self.payment_api_url,
                json=test_payload,
                timeout=5
            )
            
            api_status = "healthy" if response.status_code == 200 else f"unhealthy: HTTP {response.status_code}"
            
        except Exception as e:
            api_status = f"unhealthy: {str(e)}"
        
        # Test AI connectivity
        try:
            test_response = self.model.generate_content("Test payment justification")
            ai_status = "healthy" if test_response else "unhealthy"
        except Exception as e:
            ai_status = f"unhealthy: {str(e)}"
        
        health_status.update({
            "payment_api_status": api_status,
            "ai_model_status": ai_status,
            "payment_api_url": self.payment_api_url,
            "auto_payment_threshold": self.auto_payment_threshold,
            "manual_approval_threshold": self.manual_approval_threshold
        })
        
        return health_status