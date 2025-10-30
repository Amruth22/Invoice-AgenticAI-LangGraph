"""
Escalation Agent for Invoice Processing
Handles human-in-the-loop workflows, escalation routing, and approval management
"""

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv

from agents.base_agent import BaseAgent
from state import (
    InvoiceProcessingState, ProcessingStatus, PaymentStatus,
    RiskLevel, ValidationStatus
)
from utils.logger import StructuredLogger

load_dotenv()


class EscalationAgent(BaseAgent):
    """
    Agent responsible for escalation management and human-in-the-loop workflows
    Routes issues to appropriate human reviewers and manages approval processes
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("escalation", config)
        self.structured_logger = StructuredLogger("escalation_agent")
        
        # Initialize Gemini AI for escalation summaries
        api_key = os.getenv("GEMINI_API_KEY_1")  # Reuse first key
        if not api_key:
            raise ValueError("GEMINI_API_KEY_1 not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Escalation configuration
        self.escalation_rules = config.get("escalation_rules", {
            "high_value_threshold": 25000,
            "critical_risk_threshold": 0.8,
            "compliance_violation_escalate": True,
            "fraud_indicator_escalate": True
        })
        
        # Approval hierarchy
        self.approval_hierarchy = config.get("approval_hierarchy", {
            "supervisor": {"limit": 10000, "email": "supervisor@company.com"},
            "manager": {"limit": 50000, "email": "manager@company.com"},
            "director": {"limit": 100000, "email": "director@company.com"},
            "cfo": {"limit": float('inf'), "email": "cfo@company.com"}
        })
        
        # Notification settings
        self.email_config = config.get("email_config", {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": os.getenv("EMAIL_USERNAME"),
            "password": os.getenv("EMAIL_PASSWORD"),
            "from_address": "invoice-system@company.com"
        })
        
        # SLA settings
        self.sla_hours = config.get("sla_hours", {
            "low_priority": 48,
            "medium_priority": 24,
            "high_priority": 8,
            "critical_priority": 2
        })
        
        # Escalation tracking
        self.escalation_output_dir = config.get("escalation_output_dir", "output/escalations")
        os.makedirs(self.escalation_output_dir, exist_ok=True)
    
    def _validate_preconditions(self, state: InvoiceProcessingState) -> bool:
        """Validate that escalation is required"""
        return (
            state.escalation_required or
            state.human_review_required or
            state.should_escalate()
        )
    
    def _validate_postconditions(self, state: InvoiceProcessingState) -> bool:
        """Validate that escalation was processed"""
        return True  # Escalation agent always completes its task
    
    async def execute(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """
        Execute escalation workflow
        """
        try:
            # Step 1: Determine escalation type and priority
            escalation_type = self._determine_escalation_type(state)
            priority_level = self._calculate_priority_level(state)
            
            # Step 2: Route to appropriate approver
            approver_info = self._route_to_approver(state, escalation_type, priority_level)
            
            # Step 3: Generate escalation summary
            escalation_summary = await self._generate_escalation_summary(state, escalation_type, approver_info)
            
            # Step 4: Create escalation record
            escalation_record = await self._create_escalation_record(
                state, escalation_type, priority_level, approver_info, escalation_summary
            )
            
            # Step 5: Send notifications
            notification_result = await self._send_escalation_notifications(
                state, escalation_record, approver_info
            )
            
            # Step 6: Set up monitoring and SLA tracking
            await self._setup_sla_monitoring(state, escalation_record, priority_level)
            
            # Update state
            state.overall_status = ProcessingStatus.ESCALATED
            state.escalation_required = True
            
            # Add escalation details to business context
            state.business_context.update({
                "escalation_type": escalation_type,
                "priority_level": priority_level,
                "approver": approver_info["role"],
                "escalation_id": escalation_record["escalation_id"],
                "sla_deadline": escalation_record["sla_deadline"]
            })
            
            # Log escalation
            self.structured_logger.log_escalation(
                agent_name=self.agent_name,
                process_id=state.process_id,
                reason=f"{escalation_type} escalation to {approver_info['role']}: {state.escalation_reason}"
            )
            
            return state
            
        except Exception as e:
            self.structured_logger.log_agent_error(
                agent_name=self.agent_name,
                process_id=state.process_id,
                error=e
            )
            raise
    
    def _determine_escalation_type(self, state: InvoiceProcessingState) -> str:
        """
        Determine the type of escalation needed
        """
        if state.risk_assessment and state.risk_assessment.risk_level == RiskLevel.CRITICAL:
            return "critical_risk"
        
        if state.risk_assessment and state.risk_assessment.fraud_indicators:
            return "fraud_investigation"
        
        if state.validation_result and state.validation_result.validation_status == ValidationStatus.INVALID:
            return "validation_failure"
        
        if state.payment_decision and state.payment_decision.payment_status == PaymentStatus.REQUIRES_ESCALATION:
            return "payment_approval"
        
        if state.invoice_data and state.invoice_data.total > self.escalation_rules["high_value_threshold"]:
            return "high_value_approval"
        
        if state.human_review_required:
            return "manual_review"
        
        return "general_escalation"
    
    def _calculate_priority_level(self, state: InvoiceProcessingState) -> str:
        """
        Calculate priority level for escalation
        """
        priority_factors = []
        
        # Risk-based priority
        if state.risk_assessment:
            if state.risk_assessment.risk_level == RiskLevel.CRITICAL:
                priority_factors.append("critical")
            elif state.risk_assessment.risk_level == RiskLevel.HIGH:
                priority_factors.append("high")
        
        # Amount-based priority
        if state.invoice_data:
            if state.invoice_data.total > 100000:
                priority_factors.append("critical")
            elif state.invoice_data.total > 50000:
                priority_factors.append("high")
            elif state.invoice_data.total > 10000:
                priority_factors.append("medium")
        
        # Due date urgency
        if state.invoice_data and state.invoice_data.due_date:
            due_date = self._parse_date(state.invoice_data.due_date)
            if due_date:
                days_until_due = (due_date - datetime.now().date()).days
                if days_until_due <= 0:
                    priority_factors.append("critical")
                elif days_until_due <= 1:
                    priority_factors.append("high")
                elif days_until_due <= 3:
                    priority_factors.append("medium")
        
        # Fraud indicators
        if state.risk_assessment and state.risk_assessment.fraud_indicators:
            priority_factors.append("high")
        
        # Determine final priority
        if "critical" in priority_factors:
            return "critical"
        elif "high" in priority_factors:
            return "high"
        elif "medium" in priority_factors:
            return "medium"
        else:
            return "low"
    
    def _route_to_approver(self, state: InvoiceProcessingState, 
                          escalation_type: str, priority_level: str) -> Dict[str, Any]:
        """
        Route escalation to appropriate approver based on rules
        """
        amount = state.invoice_data.total if state.invoice_data else 0
        
        # Special routing for specific escalation types
        if escalation_type == "fraud_investigation":
            return {
                "role": "director",
                "email": self.approval_hierarchy["director"]["email"],
                "reason": "Fraud investigation requires director approval"
            }
        
        if escalation_type == "critical_risk":
            return {
                "role": "cfo",
                "email": self.approval_hierarchy["cfo"]["email"],
                "reason": "Critical risk requires CFO approval"
            }
        
        # Amount-based routing
        for role, config in self.approval_hierarchy.items():
            if amount <= config["limit"]:
                return {
                    "role": role,
                    "email": config["email"],
                    "reason": f"Amount ${amount:.2f} within {role} approval limit"
                }
        
        # Default to highest level
        return {
            "role": "cfo",
            "email": self.approval_hierarchy["cfo"]["email"],
            "reason": "Amount exceeds all approval limits"
        }
    
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
    
    async def _generate_escalation_summary(self, state: InvoiceProcessingState,
                                         escalation_type: str, approver_info: Dict[str, Any]) -> str:
        """
        Generate AI-powered escalation summary for human reviewers
        """
        context = {
            "escalation_type": escalation_type,
            "process_id": state.process_id,
            "invoice_number": state.invoice_data.invoice_number if state.invoice_data else "Unknown",
            "customer_name": state.invoice_data.customer_name if state.invoice_data else "Unknown",
            "amount": state.invoice_data.total if state.invoice_data else 0,
            "escalation_reason": state.escalation_reason,
            "risk_level": state.risk_assessment.risk_level.value if state.risk_assessment else "unknown",
            "validation_status": state.validation_result.validation_status.value if state.validation_result else "unknown",
            "approver_role": approver_info["role"]
        }
        
        prompt = f"""
You are an executive assistant preparing an escalation summary for a senior approver.

Context:
{json.dumps(context, indent=2)}

Generate a concise, professional escalation summary that includes:
1. Clear statement of what requires approval/review
2. Key risk factors and concerns
3. Financial impact and urgency
4. Recommended action
5. Any time-sensitive considerations

Keep it executive-level: clear, concise, and action-oriented.
Focus on business impact and decision-making factors.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            self.logger.warning(f"AI escalation summary failed: {e}")
            
            # Fallback to template summary
            return f"""
ESCALATION REQUIRED: {escalation_type.replace('_', ' ').title()}

Invoice: {context['invoice_number']}
Customer: {context['customer_name']}
Amount: ${context['amount']:,.2f}
Risk Level: {context['risk_level'].title()}

Reason: {context['escalation_reason']}

Action Required: Review and approve/reject this invoice payment.
Assigned to: {approver_info['role'].title()}

Please review the attached details and provide your decision.
"""
    
    async def _create_escalation_record(self, state: InvoiceProcessingState,
                                      escalation_type: str, priority_level: str,
                                      approver_info: Dict[str, Any], summary: str) -> Dict[str, Any]:
        """
        Create comprehensive escalation record
        """
        escalation_id = f"ESC_{state.process_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sla_deadline = datetime.now() + timedelta(hours=self.sla_hours[f"{priority_level}_priority"])
        
        escalation_record = {
            "escalation_id": escalation_id,
            "process_id": state.process_id,
            "escalation_type": escalation_type,
            "priority_level": priority_level,
            "created_timestamp": datetime.now().isoformat(),
            "sla_deadline": sla_deadline.isoformat(),
            
            # Approver information
            "assigned_to": approver_info["role"],
            "approver_email": approver_info["email"],
            "routing_reason": approver_info["reason"],
            
            # Invoice context
            "invoice_details": {
                "invoice_number": state.invoice_data.invoice_number if state.invoice_data else None,
                "customer_name": state.invoice_data.customer_name if state.invoice_data else None,
                "amount": state.invoice_data.total if state.invoice_data else 0,
                "due_date": state.invoice_data.due_date if state.invoice_data else None
            },
            
            # Processing context
            "processing_summary": {
                "validation_status": state.validation_result.validation_status.value if state.validation_result else None,
                "risk_level": state.risk_assessment.risk_level.value if state.risk_assessment else None,
                "risk_score": state.risk_assessment.risk_score if state.risk_assessment else 0,
                "fraud_indicators": state.risk_assessment.fraud_indicators if state.risk_assessment else [],
                "compliance_issues": state.risk_assessment.compliance_issues if state.risk_assessment else []
            },
            
            # Escalation details
            "escalation_reason": state.escalation_reason,
            "escalation_summary": summary,
            "status": "pending",
            "resolution": None,
            "resolved_timestamp": None,
            "resolver": None
        }
        
        # Save escalation record
        escalation_file = os.path.join(
            self.escalation_output_dir,
            f"escalation_{escalation_id}.json"
        )
        
        with open(escalation_file, 'w') as f:
            json.dump(escalation_record, f, indent=2, default=str)
        
        return escalation_record
    
    async def _send_escalation_notifications(self, state: InvoiceProcessingState,
                                           escalation_record: Dict[str, Any],
                                           approver_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send escalation notifications to approvers
        """
        try:
            # Prepare email content
            subject = f"URGENT: Invoice Escalation - {escalation_record['escalation_type'].replace('_', ' ').title()}"
            
            body = f"""
Dear {approver_info['role'].title()},

An invoice requires your immediate attention and approval.

{escalation_record['escalation_summary']}

Escalation Details:
- Escalation ID: {escalation_record['escalation_id']}
- Priority: {escalation_record['priority_level'].title()}
- SLA Deadline: {escalation_record['sla_deadline']}
- Invoice Number: {escalation_record['invoice_details']['invoice_number']}
- Customer: {escalation_record['invoice_details']['customer_name']}
- Amount: ${escalation_record['invoice_details']['amount']:,.2f}

Please review and provide your decision as soon as possible.

Best regards,
Invoice Processing System
"""
            
            # Send email notification
            if self.email_config.get("username") and self.email_config.get("password"):
                email_result = self._send_email(
                    to_email=approver_info["email"],
                    subject=subject,
                    body=body
                )
            else:
                email_result = {"status": "skipped", "reason": "Email credentials not configured"}
            
            # Log notification
            self.logger.info(f"Escalation notification sent to {approver_info['role']}: {email_result['status']}")
            
            return {
                "notification_sent": email_result["status"] == "sent",
                "email_result": email_result,
                "approver": approver_info["role"],
                "escalation_id": escalation_record["escalation_id"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to send escalation notification: {e}")
            return {
                "notification_sent": False,
                "error": str(e),
                "escalation_id": escalation_record["escalation_id"]
            }
    
    def _send_email(self, to_email: str, subject: str, body: str) -> Dict[str, Any]:
        """
        Send email notification
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config["from_address"]
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"])
            server.starttls()
            server.login(self.email_config["username"], self.email_config["password"])
            
            text = msg.as_string()
            server.sendmail(self.email_config["from_address"], to_email, text)
            server.quit()
            
            return {"status": "sent", "timestamp": datetime.now().isoformat()}
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _setup_sla_monitoring(self, state: InvoiceProcessingState,
                                  escalation_record: Dict[str, Any], priority_level: str):
        """
        Set up SLA monitoring for the escalation
        """
        # In a real system, this would set up monitoring/alerting
        # For now, we'll just log the SLA requirements
        
        sla_hours = self.sla_hours[f"{priority_level}_priority"]
        deadline = datetime.fromisoformat(escalation_record["sla_deadline"])
        
        self.logger.info(
            f"SLA monitoring set up for escalation {escalation_record['escalation_id']}: "
            f"{sla_hours} hours deadline at {deadline}"
        )
        
        # Add SLA tracking to state
        state.business_context["sla_monitoring"] = {
            "deadline": escalation_record["sla_deadline"],
            "hours_remaining": sla_hours,
            "priority": priority_level
        }
    
    async def resolve_escalation(self, escalation_id: str, resolution: str, 
                               resolver: str) -> Dict[str, Any]:
        """
        Resolve an escalation (called by external approval system)
        """
        try:
            escalation_file = os.path.join(self.escalation_output_dir, f"escalation_{escalation_id}.json")
            
            if not os.path.exists(escalation_file):
                raise ValueError(f"Escalation {escalation_id} not found")
            
            # Load escalation record
            with open(escalation_file, 'r') as f:
                escalation_record = json.load(f)
            
            # Update resolution
            escalation_record.update({
                "status": "resolved",
                "resolution": resolution,
                "resolved_timestamp": datetime.now().isoformat(),
                "resolver": resolver
            })
            
            # Save updated record
            with open(escalation_file, 'w') as f:
                json.dump(escalation_record, f, indent=2, default=str)
            
            self.logger.info(f"Escalation {escalation_id} resolved by {resolver}: {resolution}")
            
            return {
                "status": "success",
                "escalation_id": escalation_id,
                "resolution": resolution,
                "resolver": resolver
            }
            
        except Exception as e:
            self.logger.error(f"Failed to resolve escalation {escalation_id}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for escalation agent"""
        health_status = await super().health_check()
        
        # Check escalation directory
        try:
            test_file = os.path.join(self.escalation_output_dir, "health_check.txt")
            with open(test_file, 'w') as f:
                f.write("health check")
            os.remove(test_file)
            storage_status = "healthy"
        except Exception as e:
            storage_status = f"unhealthy: {str(e)}"
        
        # Test email configuration
        email_status = "configured" if self.email_config.get("username") else "not_configured"
        
        # Test AI connectivity
        try:
            test_response = self.model.generate_content("Test escalation summary")
            ai_status = "healthy" if test_response else "unhealthy"
        except Exception as e:
            ai_status = f"unhealthy: {str(e)}"
        
        health_status.update({
            "storage_status": storage_status,
            "email_status": email_status,
            "ai_model_status": ai_status,
            "escalation_output_dir": self.escalation_output_dir,
            "approval_hierarchy": list(self.approval_hierarchy.keys()),
            "sla_hours": self.sla_hours
        })
        
        return health_status