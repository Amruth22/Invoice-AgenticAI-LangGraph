"""
Audit Agent for Invoice Processing
Handles compliance tracking, audit trail generation, and regulatory reporting
"""

import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv

from agents.base_agent import BaseAgent
from state import (
    InvoiceProcessingState, ProcessingStatus, PaymentStatus,
    ValidationStatus, RiskLevel
)
from utils.logger import StructuredLogger

load_dotenv()


class AuditAgent(BaseAgent):
    """
    Agent responsible for audit trail generation, compliance tracking, and reporting
    Ensures all processing steps are properly documented for regulatory compliance
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("audit", config)
        self.structured_logger = StructuredLogger("audit_agent")
        
        # Initialize Gemini AI for audit summaries
        api_key = os.getenv("GEMINI_API_KEY_4")
        if not api_key:
            raise ValueError("GEMINI_API_KEY_4 not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Audit configuration
        self.audit_output_dir = config.get("audit_output_dir", "output/audit")
        self.compliance_standards = config.get("compliance_standards", [
            "SOX", "GDPR", "PCI-DSS", "ISO27001"
        ])
        
        # Retention policies
        self.retention_days = config.get("retention_days", 2555)  # 7 years default
        self.archive_threshold_days = config.get("archive_threshold_days", 365)
        
        # Reporting thresholds
        self.high_value_threshold = config.get("high_value_threshold", 25000)
        self.suspicious_activity_threshold = config.get("suspicious_activity_threshold", 0.7)
        
        # Ensure audit directory exists
        os.makedirs(self.audit_output_dir, exist_ok=True)
    
    def _validate_preconditions(self, state: InvoiceProcessingState) -> bool:
        """Validate that we have complete processing data for audit"""
        return (
            state.invoice_data is not None and
            state.validation_result is not None and
            state.risk_assessment is not None and
            state.payment_decision is not None
        )
    
    def _validate_postconditions(self, state: InvoiceProcessingState) -> bool:
        """Validate that audit records were created"""
        return len(state.audit_trail) > 0
    
    async def execute(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """
        Execute audit workflow
        """
        try:
            # Step 1: Generate comprehensive audit record
            audit_record = await self._generate_audit_record(state)
            
            # Step 2: Perform compliance checks
            compliance_results = await self._perform_compliance_checks(state, audit_record)
            
            # Step 3: Generate audit summary
            audit_summary = await self._generate_audit_summary(state, audit_record, compliance_results)
            
            # Step 4: Save audit records
            await self._save_audit_records(state, audit_record, audit_summary, compliance_results)
            
            # Step 5: Check for reportable events
            reportable_events = await self._identify_reportable_events(state, audit_record)
            
            # Step 6: Generate alerts if needed
            if reportable_events:
                await self._generate_audit_alerts(state, reportable_events)
            
            # Update state with audit completion
            state.add_audit_entry(
                agent_name=self.agent_name,
                action="audit_completed",
                status=ProcessingStatus.COMPLETED,
                details={
                    "audit_record_id": audit_record["audit_id"],
                    "compliance_status": compliance_results["overall_status"],
                    "reportable_events": len(reportable_events)
                }
            )
            
            # Log audit completion
            self.structured_logger.log_decision(
                agent_name=self.agent_name,
                process_id=state.process_id,
                decision="audit_completed",
                reasoning=f"Audit completed with {compliance_results['overall_status']} compliance status",
                confidence=1.0
            )
            
            return state
            
        except Exception as e:
            self.structured_logger.log_agent_error(
                agent_name=self.agent_name,
                process_id=state.process_id,
                error=e
            )
            raise
    
    async def _generate_audit_record(self, state: InvoiceProcessingState) -> Dict[str, Any]:
        """
        Generate comprehensive audit record for the invoice processing
        """
        audit_record = {
            "audit_id": f"AUDIT_{state.process_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "process_id": state.process_id,
            "timestamp": datetime.now().isoformat(),
            "file_name": state.file_name,
            
            # Invoice details
            "invoice_data": {
                "invoice_number": state.invoice_data.invoice_number,
                "order_id": state.invoice_data.order_id,
                "customer_name": state.invoice_data.customer_name,
                "total_amount": state.invoice_data.total,
                "due_date": state.invoice_data.due_date,
                "item_count": len(state.invoice_data.item_details),
                "extraction_confidence": state.invoice_data.extraction_confidence
            },
            
            # Validation results
            "validation_results": {
                "status": state.validation_result.validation_status.value,
                "po_found": state.validation_result.po_found,
                "discrepancies": state.validation_result.discrepancies,
                "confidence_score": state.validation_result.confidence_score
            },
            
            # Risk assessment
            "risk_assessment": {
                "risk_level": state.risk_assessment.risk_level.value,
                "risk_score": state.risk_assessment.risk_score,
                "fraud_indicators": state.risk_assessment.fraud_indicators,
                "compliance_issues": state.risk_assessment.compliance_issues,
                "recommendation": state.risk_assessment.recommendation
            },
            
            # Payment decision
            "payment_decision": {
                "status": state.payment_decision.payment_status.value,
                "approved_amount": state.payment_decision.approved_amount,
                "transaction_id": state.payment_decision.transaction_id,
                "payment_method": state.payment_decision.payment_method,
                "rejection_reason": state.payment_decision.rejection_reason,
                "approval_chain": state.payment_decision.approval_chain
            },
            
            # Processing metadata
            "processing_metadata": {
                "workflow_type": state.workflow_type,
                "overall_status": state.overall_status.value,
                "retry_count": state.retry_count,
                "escalation_required": state.escalation_required,
                "human_review_required": state.human_review_required,
                "completed_agents": state.completed_agents,
                "processing_duration_minutes": (state.updated_at - state.created_at).total_seconds() / 60
            },
            
            # Agent performance
            "agent_metrics": {
                name: {
                    "executions": metrics.executions,
                    "success_rate": metrics.successes / max(metrics.executions, 1),
                    "average_duration_ms": metrics.average_duration_ms
                }
                for name, metrics in state.agent_metrics.items()
            },
            
            # Full audit trail
            "audit_trail": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "agent_name": entry.agent_name,
                    "action": entry.action,
                    "status": entry.status.value,
                    "details": entry.details,
                    "error_message": entry.error_message
                }
                for entry in state.audit_trail
            ]
        }
        
        return audit_record
    
    async def _perform_compliance_checks(self, state: InvoiceProcessingState, 
                                       audit_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform compliance checks against various standards
        """
        compliance_results = {
            "overall_status": "compliant",
            "checks_performed": [],
            "violations": [],
            "warnings": []
        }
        
        # SOX Compliance (Sarbanes-Oxley)
        sox_result = self._check_sox_compliance(state, audit_record)
        compliance_results["checks_performed"].append("SOX")
        if sox_result["violations"]:
            compliance_results["violations"].extend(sox_result["violations"])
            compliance_results["overall_status"] = "non_compliant"
        
        # Data Privacy (GDPR-like)
        privacy_result = self._check_data_privacy_compliance(state, audit_record)
        compliance_results["checks_performed"].append("Data_Privacy")
        if privacy_result["violations"]:
            compliance_results["violations"].extend(privacy_result["violations"])
        
        # Financial Controls
        financial_result = self._check_financial_controls(state, audit_record)
        compliance_results["checks_performed"].append("Financial_Controls")
        if financial_result["violations"]:
            compliance_results["violations"].extend(financial_result["violations"])
            compliance_results["overall_status"] = "non_compliant"
        
        # Audit Trail Completeness
        audit_trail_result = self._check_audit_trail_completeness(state, audit_record)
        compliance_results["checks_performed"].append("Audit_Trail")
        if audit_trail_result["violations"]:
            compliance_results["violations"].extend(audit_trail_result["violations"])
        
        return compliance_results
    
    def _check_sox_compliance(self, state: InvoiceProcessingState, 
                            audit_record: Dict[str, Any]) -> Dict[str, List[str]]:
        """Check SOX compliance requirements"""
        violations = []
        
        # Segregation of duties
        if len(set(state.completed_agents)) < 2:
            violations.append("Insufficient segregation of duties - single agent processing")
        
        # Approval requirements for high-value transactions
        if (state.invoice_data.total > self.high_value_threshold and
            not state.human_review_required and
            state.payment_decision.payment_status == PaymentStatus.PROCESSED):
            violations.append(f"High-value transaction (${state.invoice_data.total}) processed without human approval")
        
        # Documentation requirements
        if not state.payment_decision.approval_chain:
            violations.append("Missing approval chain documentation")
        
        return {"violations": violations}
    
    def _check_data_privacy_compliance(self, state: InvoiceProcessingState,
                                     audit_record: Dict[str, Any]) -> Dict[str, List[str]]:
        """Check data privacy compliance"""
        violations = []
        
        # PII handling
        if state.invoice_data.raw_text and len(state.invoice_data.raw_text) > 10000:
            violations.append("Excessive raw text retention may contain PII")
        
        # Data retention
        if not hasattr(self, 'retention_policy_applied'):
            violations.append("Data retention policy not explicitly applied")
        
        return {"violations": violations}
    
    def _check_financial_controls(self, state: InvoiceProcessingState,
                                audit_record: Dict[str, Any]) -> Dict[str, List[str]]:
        """Check financial control compliance"""
        violations = []
        
        # Three-way matching
        if (state.validation_result.validation_status != ValidationStatus.VALID and
            state.payment_decision.payment_status == PaymentStatus.PROCESSED):
            violations.append("Payment processed without proper three-way matching")
        
        # Risk assessment requirement
        if (state.risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] and
            state.payment_decision.payment_status == PaymentStatus.PROCESSED and
            not state.human_review_required):
            violations.append("High-risk payment processed without human review")
        
        # Duplicate payment check
        if not hasattr(self, 'duplicate_check_performed'):
            violations.append("Duplicate payment check not performed")
        
        return {"violations": violations}
    
    def _check_audit_trail_completeness(self, state: InvoiceProcessingState,
                                      audit_record: Dict[str, Any]) -> Dict[str, List[str]]:
        """Check audit trail completeness"""
        violations = []
        
        # Required audit entries
        required_actions = ["started", "completed", "decision"]
        for agent in state.completed_agents:
            agent_actions = [entry.action for entry in state.audit_trail if entry.agent_name == agent]
            missing_actions = [action for action in required_actions if action not in agent_actions]
            if missing_actions:
                violations.append(f"Missing audit actions for {agent}: {missing_actions}")
        
        # Timestamp consistency
        timestamps = [entry.timestamp for entry in state.audit_trail]
        if timestamps != sorted(timestamps):
            violations.append("Audit trail timestamps are not chronological")
        
        return {"violations": violations}
    
    async def _generate_audit_summary(self, state: InvoiceProcessingState,
                                    audit_record: Dict[str, Any],
                                    compliance_results: Dict[str, Any]) -> str:
        """
        Generate AI-powered audit summary
        """
        context = {
            "process_id": state.process_id,
            "invoice_number": state.invoice_data.invoice_number,
            "customer_name": state.invoice_data.customer_name,
            "amount": state.invoice_data.total,
            "final_status": state.payment_decision.payment_status.value,
            "risk_level": state.risk_assessment.risk_level.value,
            "compliance_status": compliance_results["overall_status"],
            "violations": compliance_results["violations"],
            "processing_time": audit_record["processing_metadata"]["processing_duration_minutes"]
        }
        
        prompt = f"""
You are an audit manager generating a summary for invoice processing audit records.

Context:
{json.dumps(context, indent=2)}

Generate a concise audit summary that includes:
1. Processing outcome and key decisions
2. Compliance status and any violations
3. Risk factors and mitigation actions
4. Recommendations for process improvement

Keep the summary professional and suitable for regulatory review.
Focus on control effectiveness and compliance adherence.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            self.logger.warning(f"AI audit summary failed: {e}")
            
            # Fallback to template summary
            return f"""
Audit Summary for Invoice {state.invoice_data.invoice_number}:
- Amount: ${state.invoice_data.total:.2f}
- Customer: {state.invoice_data.customer_name}
- Final Status: {state.payment_decision.payment_status.value}
- Risk Level: {state.risk_assessment.risk_level.value}
- Compliance: {compliance_results['overall_status']}
- Processing Time: {audit_record['processing_metadata']['processing_duration_minutes']:.1f} minutes
- Violations: {len(compliance_results['violations'])}
"""
    
    async def _save_audit_records(self, state: InvoiceProcessingState,
                                audit_record: Dict[str, Any],
                                audit_summary: str,
                                compliance_results: Dict[str, Any]):
        """
        Save audit records to persistent storage
        """
        # Save detailed audit record
        audit_file = os.path.join(
            self.audit_output_dir,
            f"audit_{state.process_id}_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
        with open(audit_file, 'w') as f:
            json.dump(audit_record, f, indent=2, default=str)
        
        # Save audit summary
        summary_file = os.path.join(
            self.audit_output_dir,
            f"summary_{state.process_id}_{datetime.now().strftime('%Y%m%d')}.txt"
        )
        
        with open(summary_file, 'w') as f:
            f.write(audit_summary)
        
        # Append to daily audit log
        daily_log_file = os.path.join(
            self.audit_output_dir,
            f"daily_audit_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        
        audit_row = {
            "timestamp": datetime.now().isoformat(),
            "process_id": state.process_id,
            "invoice_number": state.invoice_data.invoice_number,
            "customer_name": state.invoice_data.customer_name,
            "amount": state.invoice_data.total,
            "final_status": state.payment_decision.payment_status.value,
            "risk_level": state.risk_assessment.risk_level.value,
            "compliance_status": compliance_results["overall_status"],
            "violations_count": len(compliance_results["violations"]),
            "processing_duration_minutes": audit_record["processing_metadata"]["processing_duration_minutes"]
        }
        
        # Append to CSV (create if doesn't exist)
        df = pd.DataFrame([audit_row])
        if os.path.exists(daily_log_file):
            df.to_csv(daily_log_file, mode='a', header=False, index=False)
        else:
            df.to_csv(daily_log_file, index=False)
        
        self.logger.info(f"Audit records saved for process {state.process_id}")
    
    async def _identify_reportable_events(self, state: InvoiceProcessingState,
                                        audit_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify events that require regulatory reporting
        """
        reportable_events = []
        
        # High-value transactions
        if state.invoice_data.total > self.high_value_threshold:
            reportable_events.append({
                "type": "high_value_transaction",
                "description": f"Transaction exceeds ${self.high_value_threshold} threshold",
                "amount": state.invoice_data.total,
                "requires_filing": True
            })
        
        # Suspicious activity
        if state.risk_assessment.risk_score > self.suspicious_activity_threshold:
            reportable_events.append({
                "type": "suspicious_activity",
                "description": f"High risk score: {state.risk_assessment.risk_score}",
                "indicators": state.risk_assessment.fraud_indicators,
                "requires_investigation": True
            })
        
        # Compliance violations
        if audit_record.get("compliance_results", {}).get("violations"):
            reportable_events.append({
                "type": "compliance_violation",
                "description": "Compliance violations detected",
                "violations": audit_record["compliance_results"]["violations"],
                "requires_remediation": True
            })
        
        # Failed payments with high amounts
        if (state.payment_decision.payment_status == PaymentStatus.FAILED and
            state.invoice_data.total > 10000):
            reportable_events.append({
                "type": "failed_high_value_payment",
                "description": "High-value payment failed",
                "amount": state.invoice_data.total,
                "reason": state.payment_decision.rejection_reason
            })
        
        return reportable_events
    
    async def _generate_audit_alerts(self, state: InvoiceProcessingState,
                                   reportable_events: List[Dict[str, Any]]):
        """
        Generate alerts for reportable events
        """
        for event in reportable_events:
            alert_message = f"AUDIT ALERT: {event['type']} - {event['description']}"
            
            self.structured_logger.log_escalation(
                agent_name=self.agent_name,
                process_id=state.process_id,
                reason=alert_message
            )
            
            # In a real system, this would send notifications to compliance team
            self.logger.warning(f"Reportable event: {alert_message}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for audit agent"""
        health_status = await super().health_check()
        
        # Check audit directory accessibility
        try:
            test_file = os.path.join(self.audit_output_dir, "health_check.txt")
            with open(test_file, 'w') as f:
                f.write("health check")
            os.remove(test_file)
            storage_status = "healthy"
        except Exception as e:
            storage_status = f"unhealthy: {str(e)}"
        
        # Test AI connectivity
        try:
            test_response = self.model.generate_content("Test audit summary")
            ai_status = "healthy" if test_response else "unhealthy"
        except Exception as e:
            ai_status = f"unhealthy: {str(e)}"
        
        health_status.update({
            "storage_status": storage_status,
            "ai_model_status": ai_status,
            "audit_output_dir": self.audit_output_dir,
            "compliance_standards": self.compliance_standards,
            "retention_days": self.retention_days
        })
        
        return health_status