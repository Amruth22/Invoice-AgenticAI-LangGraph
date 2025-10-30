"""
State Models for Invoice Processing LangGraph Workflow
Defines all state objects and data structures used across agents
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    """Processing status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"
    REQUIRES_REVIEW = "requires_review"


class ValidationStatus(str, Enum):
    """Validation status enumeration"""
    VALID = "valid"
    INVALID = "invalid"
    PARTIAL_MATCH = "partial_match"
    MISSING_PO = "missing_po"
    REQUIRES_APPROVAL = "requires_approval"


class RiskLevel(str, Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PaymentStatus(str, Enum):
    """Payment status enumeration"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    PROCESSED = "processed"
    FAILED = "failed"
    REQUIRES_ESCALATION = "requires_escalation"


class ItemDetail(BaseModel):
    """Individual item in an invoice"""
    item_name: str
    quantity: int
    rate: float
    amount: float
    category: Optional[str] = None


class InvoiceData(BaseModel):
    """Extracted invoice information"""
    invoice_number: str
    order_id: str
    customer_name: str
    due_date: Optional[str] = None
    ship_to: Optional[str] = None
    ship_mode: Optional[str] = None
    subtotal: float = 0.0
    discount: float = 0.0
    shipping_cost: float = 0.0
    total: float = 0.0
    item_details: List[ItemDetail] = []
    extraction_confidence: float = 0.0
    raw_text: Optional[str] = None


class ValidationResult(BaseModel):
    """Validation results against purchase orders"""
    po_found: bool = False
    quantity_match: bool = False
    rate_match: bool = False
    amount_match: bool = False
    validation_status: ValidationStatus = ValidationStatus.INVALID
    validation_result: str = ""
    discrepancies: List[str] = []
    confidence_score: float = 0.0
    expected_amount: Optional[float] = None
    po_data: Optional[Dict[str, Any]] = None


class RiskAssessment(BaseModel):
    """Risk assessment results"""
    risk_level: RiskLevel = RiskLevel.LOW
    risk_score: float = 0.0
    fraud_indicators: List[str] = []
    compliance_issues: List[str] = []
    recommendation: Literal["approve", "escalate", "hold", "investigate"] = "approve"
    reason: str = ""
    requires_human_review: bool = False


class PaymentDecision(BaseModel):
    """Payment processing decision"""
    payment_status: PaymentStatus = PaymentStatus.PENDING
    approved_amount: float = 0.0
    transaction_id: Optional[str] = None
    payment_method: Optional[str] = None
    approval_chain: List[str] = []
    rejection_reason: Optional[str] = None
    scheduled_date: Optional[datetime] = None


class AuditTrail(BaseModel):
    """Comprehensive audit trail"""
    process_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    agent_name: str
    action: str
    status: ProcessingStatus
    details: Dict[str, Any] = {}
    duration_ms: Optional[int] = None
    error_message: Optional[str] = None


class AgentMetrics(BaseModel):
    """Agent performance metrics"""
    agent_name: str
    executions: int = 0
    successes: int = 0
    failures: int = 0
    average_duration_ms: float = 0.0
    last_execution: Optional[datetime] = None


class InvoiceProcessingState(BaseModel):
    """Complete state object for invoice processing workflow"""
    
    # Core identifiers
    process_id: str = Field(default_factory=lambda: f"proc_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    file_name: str
    
    # Processing status
    overall_status: ProcessingStatus = ProcessingStatus.PENDING
    current_agent: Optional[str] = None
    workflow_type: str = "standard"
    
    # Document processing
    raw_text: Optional[str] = None
    invoice_data: Optional[InvoiceData] = None
    extraction_errors: List[str] = []
    
    # Validation
    validation_result: Optional[ValidationResult] = None
    validation_errors: List[str] = []
    
    # Risk assessment
    risk_assessment: Optional[RiskAssessment] = None
    risk_factors: List[str] = []
    
    # Payment processing
    payment_decision: Optional[PaymentDecision] = None
    payment_errors: List[str] = []
    
    # Audit and tracking
    audit_trail: List[AuditTrail] = []
    agent_metrics: Dict[str, AgentMetrics] = {}
    
    # Escalation and human review
    escalation_required: bool = False
    escalation_reason: Optional[str] = None
    human_review_required: bool = False
    human_review_notes: Optional[str] = None
    
    # Workflow control
    retry_count: int = 0
    max_retries: int = 3
    next_agents: List[str] = []
    completed_agents: List[str] = []
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Additional context
    customer_tier: Optional[str] = None
    priority_level: int = 1  # 1=low, 5=critical
    business_context: Dict[str, Any] = {}
    
    def add_audit_entry(self, agent_name: str, action: str, status: ProcessingStatus, 
                       details: Dict[str, Any] = None, error_message: str = None):
        """Add an audit trail entry"""
        entry = AuditTrail(
            process_id=self.process_id,
            agent_name=agent_name,
            action=action,
            status=status,
            details=details or {},
            error_message=error_message
        )
        self.audit_trail.append(entry)
        self.updated_at = datetime.now()
    
    def update_agent_metrics(self, agent_name: str, success: bool, duration_ms: int):
        """Update agent performance metrics"""
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = AgentMetrics(agent_name=agent_name)
        
        metrics = self.agent_metrics[agent_name]
        metrics.executions += 1
        metrics.last_execution = datetime.now()
        
        if success:
            metrics.successes += 1
        else:
            metrics.failures += 1
        
        # Update average duration
        if metrics.executions == 1:
            metrics.average_duration_ms = duration_ms
        else:
            metrics.average_duration_ms = (
                (metrics.average_duration_ms * (metrics.executions - 1) + duration_ms) 
                / metrics.executions
            )
    
    def should_escalate(self) -> bool:
        """Determine if workflow should be escalated"""
        return (
            self.escalation_required or
            self.retry_count >= self.max_retries or
            (self.risk_assessment and self.risk_assessment.risk_level == RiskLevel.CRITICAL) or
            (self.validation_result and self.validation_result.validation_status == ValidationStatus.REQUIRES_APPROVAL)
        )
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a summary of processing status"""
        return {
            "process_id": self.process_id,
            "file_name": self.file_name,
            "overall_status": self.overall_status.value,
            "current_agent": self.current_agent,
            "workflow_type": self.workflow_type,
            "completed_agents": self.completed_agents,
            "retry_count": self.retry_count,
            "escalation_required": self.escalation_required,
            "human_review_required": self.human_review_required,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "duration_minutes": (self.updated_at - self.created_at).total_seconds() / 60
        }


class WorkflowConfig(BaseModel):
    """Configuration for different workflow types"""
    workflow_name: str
    enabled_agents: List[str]
    routing_rules: Dict[str, Any]
    retry_policies: Dict[str, int]
    escalation_thresholds: Dict[str, Any]
    parallel_execution: bool = False
    timeout_minutes: int = 30


# Predefined workflow configurations
WORKFLOW_CONFIGS = {
    "standard": WorkflowConfig(
        workflow_name="standard",
        enabled_agents=["document", "validation", "risk", "payment", "audit"],
        routing_rules={
            "document": ["validation"],
            "validation": ["risk"],
            "risk": ["payment", "escalation"],
            "payment": ["audit"],
            "escalation": ["human_review"]
        },
        retry_policies={"document": 2, "validation": 1, "payment": 3},
        escalation_thresholds={"risk_score": 0.7, "amount": 10000}
    ),
    
    "high_value": WorkflowConfig(
        workflow_name="high_value",
        enabled_agents=["document", "validation", "risk", "audit", "escalation"],
        routing_rules={
            "document": ["validation"],
            "validation": ["risk"],
            "risk": ["audit"],
            "audit": ["escalation"],
            "escalation": ["human_review"]
        },
        retry_policies={"document": 3, "validation": 2, "payment": 2},
        escalation_thresholds={"risk_score": 0.5, "amount": 5000}
    ),
    
    "expedited": WorkflowConfig(
        workflow_name="expedited",
        enabled_agents=["document", "validation", "payment", "audit"],
        routing_rules={
            "document": ["validation"],
            "validation": ["payment"],
            "payment": ["audit"]
        },
        retry_policies={"document": 1, "validation": 1, "payment": 2},
        escalation_thresholds={"risk_score": 0.8, "amount": 1000},
        parallel_execution=True,
        timeout_minutes=10
    )
}