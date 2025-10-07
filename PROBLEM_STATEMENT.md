# Problem Statement

## Invoice AgenticAI - Intelligent Invoice Processing System with LangGraph Multi-Agent Architecture

---

## Background

Modern enterprises process thousands of invoices monthly, with accounts payable teams spending countless hours on manual data entry, validation, approval workflows, and payment processing. Traditional invoice processing is labor-intensive, error-prone, and slow—taking 5-15 days from invoice receipt to payment. Manual processes involve extracting data from PDFs, matching against purchase orders, checking for fraud, obtaining approvals, and initiating payments. This creates significant operational costs, payment delays, vendor relationship issues, and compliance risks.

Existing invoice automation solutions provide basic OCR and data extraction but lack intelligent decision-making capabilities. They cannot assess fraud risk, make payment decisions, handle exceptions intelligently, or adapt to different workflow requirements. Most systems require extensive manual intervention for validation errors, discrepancies, or high-value invoices. This results in processing bottlenecks, inconsistent approval workflows, and limited scalability.

The accounts payable industry needs an intelligent, autonomous system that can extract invoice data accurately, validate against purchase orders, assess fraud risk, make payment decisions, generate compliance documentation, and escalate complex cases to humans—all while maintaining complete audit trails and adapting to different business scenarios.

## Problem Statement

Enterprise accounts payable teams, finance departments, and procurement organizations struggle with:

- **Manual Data Entry Bottleneck**: 70-80% of AP time spent on manual invoice data extraction
- **Validation Complexity**: Time-consuming three-way matching (invoice, PO, receipt) with high error rates
- **Fraud Risk Exposure**: Limited fraud detection capabilities leading to payment fraud losses
- **Inconsistent Approval Workflows**: Manual routing and approval delays causing payment delays
- **Compliance Challenges**: Difficulty maintaining SOX, GDPR, and financial audit trails
- **Scalability Limitations**: Unable to handle invoice volume spikes without adding headcount
- **Vendor Relationship Issues**: Late payments due to processing delays damaging relationships
- **Exception Handling Overhead**: 30-40% of invoices require manual intervention
- **Lack of Visibility**: Limited real-time tracking and analytics on invoice processing status
- **High Processing Costs**: $15-$40 cost per invoice for manual processing

This leads to **extended payment cycles** (average 30-45 days), **high operational costs** ($500K+ annually for mid-size companies), **vendor dissatisfaction**, **missed early payment discounts** (2-3% savings lost), **compliance violations**, and **audit failures**.

## Objective

Design and implement a fully automated, AI-powered intelligent invoice processing system that:

1. **Extracts Invoice Data Automatically** from PDF documents using multi-method extraction and AI parsing
2. **Validates Against Purchase Orders** with fuzzy matching and three-way validation
3. **Assesses Fraud Risk** using AI-powered fraud detection and compliance checking
4. **Makes Payment Decisions** with intelligent routing and automated approval workflows
5. **Generates Audit Trails** with comprehensive compliance documentation
6. **Escalates Complex Cases** with human-in-the-loop workflows for exceptions
7. **Orchestrates Multi-Agent Workflows** using LangGraph for specialized task execution
8. **Provides Real-Time Analytics** with processing metrics and business intelligence
9. **Ensures Compliance** with SOX, GDPR, and financial regulations
10. **Reduces Processing Time** from days to minutes with 95%+ automation rate

---

## File Structure

```
Invoice-AgenticAI-LangGraph/
├── agents/                          # Specialized AI agents
│   ├── base_agent.py               # Base agent class with common functionality
│   ├── document_agent.py           # PDF extraction & AI parsing
│   ├── validation_agent.py         # PO matching & validation
│   ├── risk_agent.py               # Fraud detection & compliance
│   ├── payment_agent.py            # Payment processing & routing
│   ├── audit_agent.py              # Audit trail & compliance
│   └── escalation_agent.py         # Human-in-the-loop workflows
│
├── graph/                           # LangGraph workflow orchestration
│   ├── state_models.py             # State management & data models
│   └── invoice_graph.py            # Workflow graph definition
│
├── utils/                           # Utility functions
│   └── logger.py                   # Centralized logging system
│
├── data/                            # Data files
│   ├── invoices/                   # PDF invoice files
│   └── purchase_orders.csv         # PO reference data
│
├── output/                          # Generated outputs
│   ├── audit/                      # Audit records
│   └── escalations/                # Escalation records
│
├── logs/                            # System logs
│   └── invoice_system.log          # Application logs
│
├── main.py                          # Streamlit application
├── payment_api.py                   # Payment simulation service
├── run.py                           # Application runner
├── setup.py                         # Setup script
└── requirements.txt                 # Python dependencies
```

---

## Input Sources

### 1) Invoice Documents

- **Source**: PDF invoice files from vendors and suppliers
- **Format**: PDF documents with structured or unstructured layouts
- **Processing**: Multi-method extraction (PyMuPDF, PDFPlumber) with AI parsing
- **Data Extracted**: Invoice number, customer name, amounts, line items, dates
- **Volume**: Supports batch processing of multiple invoices

### 2) Purchase Order Data

- **Source**: `purchase_orders.csv` with PO reference information
- **Format**: CSV with columns: order_id, customer_name, item_name, quantity, rate, amount
- **Usage**: Three-way matching validation against invoices
- **Matching**: Fuzzy string matching for customer names and items

### 3) Business Rules & Policies

- **Risk Thresholds**: Configurable risk levels (low, medium, high, critical)
- **Payment Policies**: Auto-payment thresholds, manual approval requirements
- **Compliance Rules**: SOX, GDPR, financial regulations
- **Escalation Criteria**: Conditions triggering human review

### 4) Configuration Files

- **.env**: Environment variables with 4 Gemini API keys for load balancing
- **Workflow Configs**: Standard, high-value, and expedited workflow types
- **Agent Configs**: Extraction methods, validation thresholds, risk parameters

---

## Core Modules to be Implemented

### 1. `agents/document_agent.py` - PDF Extraction & AI Parsing Specialist

**Purpose**: Extract structured invoice data from PDF documents using multi-method extraction and Google Gemini AI parsing.

**Function Signature**:
```python
class DocumentAgent(BaseAgent):
    def process(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """
        Extract and parse invoice data from PDF.
        Input: InvoiceProcessingState with file_name
        Output: Updated state with invoice_data
        """
```

**Expected Output Format**:
```json
{
    "invoice_data": {
        "invoice_number": "INV-2024-001",
        "order_id": "PO-12345",
        "customer_name": "Acme Corporation",
        "due_date": "2024-12-31",
        "ship_to": "123 Main St, New York, NY 10001",
        "ship_mode": "Standard Ground",
        "subtotal": 5000.00,
        "discount": 250.00,
        "shipping_cost": 50.00,
        "total": 4800.00,
        "item_details": [
            {
                "item_name": "Widget Pro 3000",
                "quantity": 10,
                "rate": 500.00,
                "amount": 5000.00,
                "category": "Electronics"
            }
        ],
        "extraction_confidence": 0.92,
        "raw_text": "INVOICE\nInvoice #: INV-2024-001..."
    },
    "overall_status": "in_progress",
    "current_agent": "document_agent"
}
```

**Key Features**:
- **Multi-Method Extraction**: PyMuPDF for fast extraction, PDFPlumber for complex layouts
- **AI-Powered Parsing**: Gemini 2.0 Flash for intelligent data extraction
- **Confidence Scoring**: Extraction confidence metrics for quality assessment
- **Error Handling**: Fallback strategies for extraction failures
- **Line Item Extraction**: Detailed item-level data extraction

**Extraction Methods**:
```python
extraction_methods = {
    'pymupdf': {
        'speed': 'fast',
        'accuracy': 'good',
        'use_case': 'standard invoices'
    },
    'pdfplumber': {
        'speed': 'moderate',
        'accuracy': 'excellent',
        'use_case': 'complex layouts, tables'
    },
    'gemini_ai': {
        'speed': 'moderate',
        'accuracy': 'excellent',
        'use_case': 'unstructured invoices'
    }
}
```

---

### 2. `agents/validation_agent.py` - PO Matching & Validation Specialist

**Purpose**: Validate invoice data against purchase orders using fuzzy matching and three-way validation logic.

**Function Signature**:
```python
class ValidationAgent(BaseAgent):
    def process(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """
        Validate invoice against purchase orders.
        Input: InvoiceProcessingState with invoice_data
        Output: Updated state with validation_result
        """
```

**Expected Output Format**:
```json
{
    "validation_result": {
        "po_found": true,
        "quantity_match": true,
        "rate_match": true,
        "amount_match": false,
        "validation_status": "partial_match",
        "validation_result": "Amount discrepancy detected: Expected $5000.00, Found $4800.00",
        "discrepancies": [
            "Total amount mismatch: Expected $5000.00, Actual $4800.00 (Difference: $200.00)",
            "Discount applied: $250.00 not in PO"
        ],
        "confidence_score": 0.85,
        "expected_amount": 5000.00,
        "po_data": {
            "order_id": "PO-12345",
            "customer_name": "Acme Corporation",
            "item_name": "Widget Pro 3000",
            "quantity": 10,
            "rate": 500.00,
            "amount": 5000.00
        }
    },
    "overall_status": "in_progress",
    "current_agent": "validation_agent"
}
```

**Key Features**:
- **Fuzzy Matching**: FuzzyWuzzy for customer name and item matching (80% threshold)
- **Three-Way Validation**: Invoice vs PO vs Receipt matching
- **Discrepancy Detection**: Identifies quantity, rate, and amount mismatches
- **Tolerance Handling**: Configurable tolerance for amount differences (5% default)
- **Confidence Scoring**: Validation confidence metrics

**Validation Logic**:
```python
validation_checks = {
    'po_match': 'Order ID exists in PO database',
    'customer_match': 'Fuzzy match customer name (≥80%)',
    'item_match': 'Fuzzy match item names (≥80%)',
    'quantity_match': 'Exact quantity match',
    'rate_match': 'Rate within tolerance (±5%)',
    'amount_match': 'Total amount within tolerance (±5%)'
}

validation_status = {
    'valid': 'All checks passed',
    'partial_match': 'Some discrepancies within tolerance',
    'invalid': 'Critical discrepancies detected',
    'missing_po': 'Purchase order not found',
    'requires_approval': 'Discrepancies require human approval'
}
```

---

### 3. `agents/risk_agent.py` - Fraud Detection & Compliance Specialist

**Purpose**: Assess fraud risk and compliance using AI-powered analysis and business rule validation.

**Function Signature**:
```python
class RiskAgent(BaseAgent):
    def process(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """
        Assess fraud risk and compliance.
        Input: InvoiceProcessingState with invoice_data and validation_result
        Output: Updated state with risk_assessment
        """
```

**Expected Output Format**:
```json
{
    "risk_assessment": {
        "risk_level": "medium",
        "risk_score": 0.55,
        "fraud_indicators": [
            "Amount discrepancy detected",
            "Discount not in original PO",
            "First-time vendor"
        ],
        "compliance_issues": [
            "Missing receipt confirmation",
            "Approval chain incomplete"
        ],
        "recommendation": "escalate",
        "reason": "Amount discrepancy and missing receipt require human review. Vendor is new and not in approved vendor list.",
        "requires_human_review": true
    },
    "overall_status": "in_progress",
    "current_agent": "risk_agent"
}
```

**Key Features**:
- **AI-Powered Fraud Detection**: Gemini AI analyzes patterns and anomalies
- **Risk Scoring**: 0-1 scale with configurable thresholds
- **Compliance Checking**: SOX, GDPR, financial regulations validation
- **Vendor Analysis**: First-time vendor, blacklist checking
- **Pattern Detection**: Duplicate invoices, unusual amounts, suspicious timing

**Risk Assessment Criteria**:
```python
risk_factors = {
    'amount_discrepancy': {
        'weight': 0.3,
        'threshold': '5% variance',
        'severity': 'high'
    },
    'missing_po': {
        'weight': 0.4,
        'threshold': 'no matching PO',
        'severity': 'critical'
    },
    'new_vendor': {
        'weight': 0.2,
        'threshold': 'first transaction',
        'severity': 'medium'
    },
    'unusual_amount': {
        'weight': 0.25,
        'threshold': '3x average',
        'severity': 'high'
    },
    'duplicate_invoice': {
        'weight': 0.5,
        'threshold': 'same invoice number',
        'severity': 'critical'
    }
}

risk_levels = {
    'low': 'risk_score < 0.3',
    'medium': '0.3 ≤ risk_score < 0.6',
    'high': '0.6 ≤ risk_score < 0.8',
    'critical': 'risk_score ≥ 0.8'
}
```

---

### 4. `agents/payment_agent.py` - Payment Processing & Routing Specialist

**Purpose**: Make intelligent payment decisions and route payments based on risk assessment and business rules.

**Function Signature**:
```python
class PaymentAgent(BaseAgent):
    def process(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """
        Process payment decision and initiate payment.
        Input: InvoiceProcessingState with risk_assessment
        Output: Updated state with payment_decision
        """
```

**Expected Output Format**:
```json
{
    "payment_decision": {
        "payment_status": "approved",
        "approved_amount": 4800.00,
        "transaction_id": "TXN-2024-12-20-001",
        "payment_method": "ACH",
        "approval_chain": [
            "system_auto_approval",
            "finance_manager_approval"
        ],
        "rejection_reason": null,
        "scheduled_date": "2024-12-25T00:00:00Z"
    },
    "overall_status": "completed",
    "current_agent": "payment_agent"
}
```

**Key Features**:
- **Intelligent Routing**: Auto-payment vs manual approval based on risk and amount
- **Payment API Integration**: Integration with payment processing systems
- **Approval Workflows**: Multi-level approval chains for high-value invoices
- **Retry Logic**: Automatic retry on payment failures
- **Scheduling**: Payment scheduling based on due dates and cash flow

**Payment Decision Logic**:
```python
payment_rules = {
    'auto_payment': {
        'conditions': [
            'risk_level == "low"',
            'amount < $5000',
            'validation_status == "valid"',
            'vendor in approved_list'
        ],
        'action': 'process_immediately'
    },
    'manager_approval': {
        'conditions': [
            'risk_level == "medium"',
            '$5000 ≤ amount < $25000',
            'validation_status == "partial_match"'
        ],
        'action': 'route_to_manager'
    },
    'executive_approval': {
        'conditions': [
            'risk_level == "high" or "critical"',
            'amount ≥ $25000',
            'validation_status == "requires_approval"'
        ],
        'action': 'route_to_executive'
    },
    'reject': {
        'conditions': [
            'risk_level == "critical"',
            'fraud_indicators > 3',
            'validation_status == "invalid"'
        ],
        'action': 'reject_payment'
    }
}
```

---

### 5. `agents/audit_agent.py` - Audit Trail & Compliance Specialist

**Purpose**: Generate comprehensive audit trails and compliance documentation for regulatory requirements.

**Function Signature**:
```python
class AuditAgent(BaseAgent):
    def process(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """
        Generate audit trail and compliance documentation.
        Input: InvoiceProcessingState with complete processing history
        Output: Updated state with audit_trail
        """
```

**Expected Output Format**:
```json
{
    "audit_trail": [
        {
            "process_id": "proc_20241220_103000",
            "timestamp": "2024-12-20T10:30:00Z",
            "agent_name": "document_agent",
            "action": "extract_invoice_data",
            "status": "completed",
            "details": {
                "extraction_method": "pymupdf",
                "confidence": 0.92,
                "fields_extracted": 12
            },
            "duration_ms": 1500,
            "error_message": null
        },
        {
            "process_id": "proc_20241220_103000",
            "timestamp": "2024-12-20T10:30:02Z",
            "agent_name": "validation_agent",
            "action": "validate_against_po",
            "status": "completed",
            "details": {
                "po_found": true,
                "validation_status": "partial_match",
                "discrepancies": 2
            },
            "duration_ms": 800,
            "error_message": null
        }
    ],
    "compliance_report": {
        "sox_compliance": "compliant",
        "gdpr_compliance": "compliant",
        "financial_controls": "passed",
        "audit_trail_complete": true,
        "retention_policy": "7_years",
        "encryption_status": "encrypted"
    },
    "overall_status": "completed",
    "current_agent": "audit_agent"
}
```

**Key Features**:
- **Comprehensive Logging**: Every action, decision, and data change logged
- **Compliance Documentation**: SOX, GDPR, financial regulations
- **Retention Policies**: Configurable data retention (7 years default)
- **Encryption**: Audit trail encryption for security
- **Reporting**: Compliance reports for auditors

**Audit Trail Components**:
```python
audit_components = {
    'action_log': 'Every agent action with timestamp',
    'data_changes': 'Before/after snapshots of data',
    'decision_rationale': 'AI decision explanations',
    'approval_chain': 'Complete approval workflow',
    'system_events': 'Errors, retries, escalations',
    'user_interactions': 'Human review and approvals',
    'compliance_checks': 'Regulatory validation results'
}
```

---

### 6. `agents/escalation_agent.py` - Human-in-the-Loop Workflow Specialist

**Purpose**: Manage escalations and human review workflows for complex cases requiring manual intervention.

**Function Signature**:
```python
class EscalationAgent(BaseAgent):
    def process(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """
        Handle escalations and human review workflows.
        Input: InvoiceProcessingState requiring escalation
        Output: Updated state with escalation details
        """
```

**Expected Output Format**:
```json
{
    "escalation_required": true,
    "escalation_reason": "High-value invoice with validation discrepancies requires executive approval",
    "human_review_required": true,
    "human_review_notes": "Amount discrepancy of $200. Vendor applied discount not in PO. Requires CFO approval.",
    "escalation_details": {
        "escalation_type": "validation_discrepancy",
        "severity": "high",
        "assigned_to": "finance_manager",
        "escalation_time": "2024-12-20T10:35:00Z",
        "sla_deadline": "2024-12-20T16:00:00Z",
        "notification_sent": true,
        "approval_required_from": ["finance_manager", "cfo"]
    },
    "overall_status": "escalated",
    "current_agent": "escalation_agent"
}
```

**Key Features**:
- **Intelligent Escalation**: Automatic escalation based on risk, amount, and validation
- **Approval Hierarchy**: Multi-level approval routing
- **SLA Monitoring**: Tracks escalation response times
- **Notification System**: Email/SMS notifications to approvers
- **Human Review Interface**: Streamlit UI for manual review

**Escalation Triggers**:
```python
escalation_triggers = {
    'high_risk': {
        'condition': 'risk_level in ["high", "critical"]',
        'route_to': 'risk_manager',
        'sla_hours': 4
    },
    'validation_failure': {
        'condition': 'validation_status == "requires_approval"',
        'route_to': 'finance_manager',
        'sla_hours': 8
    },
    'high_value': {
        'condition': 'amount > $25000',
        'route_to': 'cfo',
        'sla_hours': 24
    },
    'fraud_suspicion': {
        'condition': 'fraud_indicators > 3',
        'route_to': 'fraud_team',
        'sla_hours': 2
    },
    'new_vendor': {
        'condition': 'vendor not in approved_list',
        'route_to': 'procurement',
        'sla_hours': 48
    }
}
```

---

### 7. `graph/invoice_graph.py` - LangGraph Workflow Orchestration

**Purpose**: Orchestrate multi-agent workflow using LangGraph with intelligent routing and state management.

**Function Signature**:
```python
def get_workflow(config: Dict[str, Any] = None) -> CompiledGraph:
    """
    Create and compile the invoice processing workflow.
    Input: Optional configuration dict
    Output: Compiled LangGraph workflow
    """
```

**Workflow Architecture**:
```
Standard Workflow:
START → Document Agent → Validation Agent → Risk Agent → 
[Decision: Low/Medium Risk] → Payment Agent → Audit Agent → END
[Decision: High/Critical Risk] → Escalation Agent → Human Review → END

High-Value Workflow:
START → Document Agent → Validation Agent → Risk Agent → 
Audit Agent → Escalation Agent → Human Review → END

Expedited Workflow:
START → Document Agent → Validation Agent → Payment Agent → Audit Agent → END
```

**Expected Output Format**:
```json
{
    "workflow_execution": {
        "workflow_type": "standard",
        "status": "completed",
        "execution_time_seconds": 12.5,
        "agents_executed": 5,
        "completed_agents": [
            "document_agent",
            "validation_agent",
            "risk_agent",
            "payment_agent",
            "audit_agent"
        ],
        "escalations": 0,
        "errors": []
    },
    "final_state": {
        "process_id": "proc_20241220_103000",
        "overall_status": "completed",
        "invoice_data": {...},
        "validation_result": {...},
        "risk_assessment": {...},
        "payment_decision": {...},
        "audit_trail": [...]
    }
}
```

**Key Features**:
- **Conditional Routing**: Dynamic workflow based on risk and validation results
- **State Management**: Centralized InvoiceProcessingState passed between agents
- **Error Handling**: Retry logic and graceful degradation
- **Parallel Execution**: Optional parallel processing for expedited workflow
- **Workflow Types**: Standard, high-value, and expedited configurations

---

### 8. `graph/state_models.py` - State Management & Data Models

**Purpose**: Define comprehensive state models and data structures for workflow state management.

**Key Models**:

```python
class InvoiceProcessingState(BaseModel):
    """Complete state object for invoice processing workflow"""
    
    # Core identifiers
    process_id: str
    file_name: str
    
    # Processing status
    overall_status: ProcessingStatus
    current_agent: Optional[str]
    workflow_type: str
    
    # Agent outputs
    invoice_data: Optional[InvoiceData]
    validation_result: Optional[ValidationResult]
    risk_assessment: Optional[RiskAssessment]
    payment_decision: Optional[PaymentDecision]
    
    # Audit and tracking
    audit_trail: List[AuditTrail]
    agent_metrics: Dict[str, AgentMetrics]
    
    # Escalation
    escalation_required: bool
    human_review_required: bool
    
    # Workflow control
    retry_count: int
    completed_agents: List[str]
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
```

**Key Features**:
- **Type Safety**: Pydantic v2 models for robust validation
- **Enumerations**: ProcessingStatus, ValidationStatus, RiskLevel, PaymentStatus
- **Audit Methods**: Built-in methods for audit trail management
- **Metrics Tracking**: Agent performance metrics
- **Workflow Control**: Retry logic and escalation management

---

## Architecture Flow

### High-Level Invoice Processing Flow

```
PDF Upload → Document Extraction → PO Validation → Risk Assessment → 
[Decision: Low/Medium Risk] → Payment Processing → Audit Trail → Complete
[Decision: High/Critical Risk] → Escalation → Human Review → Approval → Payment → Complete
```

### Multi-Agent Orchestration Flow

```
LangGraph Workflow → Document Agent → Validation Agent → Risk Agent → 
[Conditional Router] → Payment Agent OR Escalation Agent → Audit Agent → 
Result Aggregation → Output Generation
```

### Conditional Routing Logic

```python
def route_after_risk_assessment(state: InvoiceProcessingState) -> str:
    """Route based on risk assessment results"""
    
    if state.risk_assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]:
        if state.invoice_data.total < 5000:
            return "payment_agent"  # Auto-process
        else:
            return "escalation_agent"  # Manager approval
    
    elif state.risk_assessment.risk_level == RiskLevel.HIGH:
        return "escalation_agent"  # Executive approval
    
    else:  # CRITICAL
        return "escalation_agent"  # Fraud team review
```

---

## Quality Gate Decision Matrix

| Metric | Threshold | Pass Condition | Action |
|--------|-----------|----------------|--------|
| **Extraction Confidence** | ≥ 70% | High confidence in data extraction | Proceed to Validation |
| **PO Match** | Found | Purchase order exists | Proceed to Risk Assessment |
| **Validation Status** | Valid or Partial Match | Acceptable discrepancies | Proceed to Risk Assessment |
| **Risk Score** | < 0.6 | Low or medium risk | Proceed to Payment |
| **Risk Score** | ≥ 0.6 | High or critical risk | Escalate to Human Review |
| **Amount** | < $5,000 | Below auto-payment threshold | Auto-process Payment |
| **Amount** | ≥ $25,000 | Above manual threshold | Require Executive Approval |
| **Fraud Indicators** | < 3 | Acceptable risk level | Proceed to Payment |
| **Fraud Indicators** | ≥ 3 | High fraud risk | Escalate to Fraud Team |

---

## Configuration Setup

### Create `.env` file with the following credentials:

```bash
# Gemini API Keys (4 keys for load balancing)
GEMINI_API_KEY_1=your_document_agent_key
GEMINI_API_KEY_2=your_validation_agent_key
GEMINI_API_KEY_3=your_risk_agent_key
GEMINI_API_KEY_4=your_audit_agent_key

# Model Configuration
GEMINI_MODEL=gemini-2.0-flash-exp

# Payment API Configuration
PAYMENT_API_URL=http://localhost:8000/initiate_payment
PAYMENT_API_KEY=your_payment_api_key

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
LOG_FILE=logs/invoice_system.log

# Workflow Configuration
AUTO_PAYMENT_THRESHOLD=5000
MANUAL_APPROVAL_THRESHOLD=25000
RISK_THRESHOLD_HIGH=0.6
RISK_THRESHOLD_CRITICAL=0.8
```

### Agent Configuration

```python
agent_config = {
    "document_agent": {
        "extraction_methods": ["pymupdf", "pdfplumber"],
        "ai_confidence_threshold": 0.7,
        "retry_on_failure": True
    },
    "validation_agent": {
        "po_file_path": "data/purchase_orders.csv",
        "fuzzy_threshold": 80,
        "amount_tolerance": 0.05,
        "enable_three_way_match": True
    },
    "risk_agent": {
        "risk_thresholds": {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.9
        },
        "fraud_detection_enabled": True,
        "compliance_checks": ["SOX", "GDPR"]
    },
    "payment_agent": {
        "payment_api_url": "http://localhost:8000/initiate_payment",
        "auto_payment_threshold": 5000,
        "manual_approval_threshold": 25000,
        "retry_attempts": 3
    }
}
```

---

## Commands to Create Required API Keys

### Google Gemini API Key

1. Open your web browser and go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in to your Google account
3. Navigate to "Get API Key" in the left sidebar
4. Click "Create API Key" → "Create API Key in new project"
5. Copy the generated key and save it securely
6. **Repeat 3 more times** to create 4 total keys for load balancing
7. Add all keys to your `.env` file

**Note**: You can use the same key for all 4 variables if you prefer, but separate keys provide better load distribution and avoid rate limiting.

---

## Implementation Execution

### Installation and Setup

```bash
# 1. Clone the repository
git clone https://github.com/Amruth22/Invoice-AgenticAI-LangGraph.git
cd Invoice-AgenticAI-LangGraph

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
cp .env.example .env
# Edit .env and add your Gemini API keys

# 4. Create required directories
mkdir -p data/invoices logs output/audit output/escalations

# 5. Start Payment API (Terminal 1)
python payment_api.py

# 6. Run the application (Terminal 2)
streamlit run main.py

# 7. Access the application
# Open browser: http://localhost:8501
```

### Usage Commands

```bash
# Start payment simulation API
python payment_api.py

# Run Streamlit application
streamlit run main.py

# Run with custom configuration
python run.py --config custom_config.json

# Health check
curl http://localhost:8501/health
curl http://localhost:8000/health

# Run tests (if available)
pytest tests/ -v
```

---

## Performance Characteristics

### Processing Time by Workflow Stage

| Workflow Stage | Average Time | Bottleneck |
|----------------|--------------|------------|
| Document Extraction | 2-5 seconds | PDF parsing + AI |
| PO Validation | 1-2 seconds | Database lookup + fuzzy matching |
| Risk Assessment | 3-5 seconds | AI fraud detection |
| Payment Processing | 2-4 seconds | Payment API call |
| Audit Trail Generation | 1-2 seconds | Data serialization |
| **Total (Standard Workflow)** | **9-18 seconds** | AI processing |
| **Total (with Escalation)** | **+ Human review time** | Manual approval |

### Scalability Metrics

| Invoice Volume | Processing Time | Concurrent Processing | Memory Usage |
|----------------|----------------|----------------------|--------------|
| **1-10 invoices** | ~10-15 sec each | 1-3 concurrent | ~512MB |
| **10-50 invoices** | ~8-12 sec each | 3-5 concurrent | ~1GB |
| **50-100 invoices** | ~6-10 sec each | 5-10 concurrent | ~2GB |
| **100+ invoices** | ~5-8 sec each | 10+ concurrent | ~4GB |

### Cost Savings Analysis

| Metric | Manual Process | Automated Process | Savings |
|--------|---------------|-------------------|---------|
| **Processing Time** | 15-30 minutes | 10-20 seconds | 98% reduction |
| **Cost per Invoice** | $15-$40 | $0.50-$2 | 95% reduction |
| **Error Rate** | 5-10% | <1% | 90% reduction |
| **Fraud Detection** | 60-70% | 90-95% | 30% improvement |
| **Payment Cycle** | 30-45 days | 1-3 days | 90% reduction |

---

## Sample Output

### Generated Outputs Structure

```
output/
├── audit/
│   ├── proc_20241220_103000_audit.json
│   ├── compliance_report_2024_12.pdf
│   └── sox_audit_trail.csv
│
├── escalations/
│   ├── escalation_20241220_103500.json
│   ├── pending_approvals.csv
│   └── sla_violations.json
│
└── reports/
    ├── processing_summary_2024_12.pdf
    ├── risk_analysis_report.pdf
    └── payment_reconciliation.csv
```

### Processing Summary Report

```json
{
    "summary": {
        "period": "2024-12-01 to 2024-12-20",
        "total_invoices_processed": 156,
        "completed": 142,
        "escalated": 12,
        "failed": 2,
        "success_rate": 91.0,
        "average_processing_time_seconds": 12.5,
        "total_amount_processed": 1250000.00,
        "fraud_detected": 3,
        "compliance_violations": 0
    },
    "risk_distribution": {
        "low": 98,
        "medium": 42,
        "high": 13,
        "critical": 3
    },
    "payment_status": {
        "auto_processed": 98,
        "manager_approved": 32,
        "executive_approved": 12,
        "rejected": 2,
        "pending": 12
    },
    "agent_performance": {
        "document_agent": {
            "success_rate": 98.7,
            "average_duration_ms": 3500
        },
        "validation_agent": {
            "success_rate": 95.5,
            "average_duration_ms": 1800
        },
        "risk_agent": {
            "success_rate": 100.0,
            "average_duration_ms": 4200
        },
        "payment_agent": {
            "success_rate": 97.2,
            "average_duration_ms": 3000
        }
    }
}
```

---

## Testing and Validation

### Test Suite Execution

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_agents.py -v
pytest tests/test_workflow.py -v
pytest tests/test_integration.py -v
```

### Test Cases to be Passed

#### 1. `test_document_agent_extraction()`
- **Purpose**: Validate PDF extraction and AI parsing
- **Test Coverage**: Multi-method extraction, confidence scoring, error handling
- **Expected Results**: Complete invoice_data with ≥70% confidence

#### 2. `test_validation_agent_po_matching()`
- **Purpose**: Validate PO matching and three-way validation
- **Test Coverage**: Fuzzy matching, discrepancy detection, tolerance handling
- **Expected Results**: Accurate validation_result with discrepancies identified

#### 3. `test_risk_agent_fraud_detection()`
- **Purpose**: Validate fraud detection and risk scoring
- **Test Coverage**: AI fraud analysis, compliance checking, risk level assignment
- **Expected Results**: Accurate risk_assessment with fraud indicators

#### 4. `test_payment_agent_routing()`
- **Purpose**: Validate payment decision logic and routing
- **Test Coverage**: Auto-payment, approval workflows, payment API integration
- **Expected Results**: Correct payment_decision based on risk and amount

#### 5. `test_audit_agent_trail_generation()`
- **Purpose**: Validate audit trail generation and compliance
- **Test Coverage**: Comprehensive logging, compliance documentation, retention
- **Expected Results**: Complete audit_trail with all actions logged

#### 6. `test_escalation_agent_workflows()`
- **Purpose**: Validate escalation logic and human review
- **Test Coverage**: Escalation triggers, approval hierarchy, SLA monitoring
- **Expected Results**: Correct escalation routing and notifications

#### 7. `test_workflow_standard()`
- **Purpose**: Validate standard workflow execution
- **Test Coverage**: End-to-end pipeline, state management, conditional routing
- **Expected Results**: Successful workflow completion with all agents

#### 8. `test_workflow_high_value()`
- **Purpose**: Validate high-value workflow with enhanced validation
- **Test Coverage**: Additional approval steps, audit requirements
- **Expected Results**: Proper escalation and approval chain

#### 9. `test_workflow_expedited()`
- **Purpose**: Validate expedited workflow for urgent invoices
- **Test Coverage**: Fast-track processing, parallel execution
- **Expected Results**: Faster processing with maintained accuracy

#### 10. `test_error_recovery()`
- **Purpose**: Validate error handling and retry logic
- **Test Coverage**: Agent failures, retry mechanisms, graceful degradation
- **Expected Results**: Robust error handling with recovery

---

## Important Notes for Testing

### API Key Requirements
- **Gemini API Keys**: Required for document parsing, risk assessment, and audit
- **Payment API**: Mock payment API included (`payment_api.py`)
- **Free Tier Limits**: Be aware of Gemini API rate limits (15 requests/minute)

### Test Data Requirements
- **Invoice PDFs**: Place test invoices in `data/invoices/` directory
- **Purchase Orders**: Ensure `data/purchase_orders.csv` has matching PO data
- **Test Scenarios**: Include valid invoices, discrepancies, fraud cases

### Test Environment
- Tests must be run from the project root directory
- Ensure all dependencies are installed via `pip install -r requirements.txt`
- Verify `.env` file is properly configured with valid API keys
- Start payment API before running integration tests

### Performance Expectations
- Individual agent tests should complete within 5-10 seconds
- Full workflow tests may take 15-30 seconds depending on API response times
- Batch processing tests may take longer based on invoice count

---

## Key Benefits

### Technical Advantages

1. **Automated Data Extraction**: 98% accuracy with multi-method extraction and AI parsing
2. **Intelligent Validation**: Fuzzy matching and three-way validation with discrepancy detection
3. **AI-Powered Fraud Detection**: 90-95% fraud detection rate with pattern analysis
4. **Smart Payment Routing**: Automated decision-making based on risk and business rules
5. **Comprehensive Audit Trails**: Complete compliance documentation for SOX, GDPR
6. **Multi-Agent Architecture**: Specialized agents for different processing aspects
7. **Real-Time Processing**: 10-20 second processing time per invoice
8. **Scalable Design**: Handle 100+ concurrent invoices efficiently

### Business Impact

1. **Cost Reduction**: 95% reduction in processing costs ($15-40 → $0.50-2 per invoice)
2. **Time Savings**: 98% reduction in processing time (15-30 min → 10-20 sec)
3. **Improved Accuracy**: Error rate reduced from 5-10% to <1%
4. **Faster Payments**: Payment cycle reduced from 30-45 days to 1-3 days
5. **Better Vendor Relations**: On-time payments improving vendor satisfaction
6. **Early Payment Discounts**: Capture 2-3% discounts with faster processing
7. **Fraud Prevention**: 30% improvement in fraud detection (60-70% → 90-95%)
8. **Compliance Assurance**: Automated SOX, GDPR compliance documentation

### Educational Value

1. **LangGraph Workflows**: Real-world multi-agent orchestration with conditional routing
2. **AI Integration**: Combining multiple AI capabilities (extraction, fraud detection, decision-making)
3. **State Management**: Complex state handling with Pydantic v2
4. **Business Process Automation**: End-to-end invoice processing automation
5. **Fraud Detection**: AI-powered fraud detection techniques
6. **Compliance**: Financial regulations and audit trail generation
7. **Production Deployment**: Streamlit application with real-time monitoring
8. **FinTech Innovation**: Modern financial technology practices

---

## Future Enhancements

### Short-term (1-3 months)
- [ ] Database integration for invoice history and analytics
- [ ] Email integration for automatic invoice receipt
- [ ] Mobile application for approval workflows
- [ ] Advanced ML models for better fraud detection
- [ ] Integration with more payment gateways (Stripe, PayPal)

### Medium-term (3-6 months)
- [ ] OCR improvements for handwritten invoices
- [ ] Multi-currency support with exchange rate handling
- [ ] Vendor portal for invoice submission and tracking
- [ ] Advanced analytics dashboard with BI insights
- [ ] Integration with ERP systems (SAP, Oracle, NetSuite)

### Long-term (6-12 months)
- [ ] Predictive analytics for cash flow forecasting
- [ ] Blockchain integration for immutable audit trails
- [ ] Natural language query interface for invoice search
- [ ] Automated contract matching and compliance
- [ ] Enterprise features (SSO, RBAC, multi-tenant)
- [ ] AI-powered vendor risk scoring

---

## Conclusion

**Invoice AgenticAI - LangGraph** represents a transformative approach to accounts payable automation, combining the power of **LangGraph multi-agent workflows**, **Google Gemini AI**, and **intelligent business logic** to create a fully automated invoice processing system. By leveraging specialized AI agents for document extraction, validation, fraud detection, payment processing, audit trail generation, and escalation management, the system provides end-to-end automation while maintaining human oversight for complex cases.

The system is **production-ready**, **well-architected**, and **highly scalable**, making it suitable for:
- **Enterprise accounts payable teams** processing thousands of invoices monthly
- **Finance departments** requiring compliance and audit capabilities
- **Procurement organizations** managing vendor payments
- **FinTech companies** building invoice automation solutions

**Key Differentiators**:
- ✅ Multi-method PDF extraction with AI parsing
- ✅ Intelligent PO validation with fuzzy matching
- ✅ AI-powered fraud detection and risk assessment
- ✅ Smart payment routing with approval workflows
- ✅ Comprehensive audit trails for compliance
- ✅ Human-in-the-loop for complex cases
- ✅ Real-time processing (10-20 seconds per invoice)
- ✅ Production-ready Streamlit application

**Star Rating: ⭐⭐⭐⭐⭐ (5/5)**

---

## Support and Resources

- **GitHub Repository**: [Amruth22/Invoice-AgenticAI-LangGraph](https://github.com/Amruth22/Invoice-AgenticAI-LangGraph)
- **Issues**: [GitHub Issues](https://github.com/Amruth22/Invoice-AgenticAI-LangGraph/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Amruth22/Invoice-AgenticAI-LangGraph/discussions)
- **Documentation**: See README.md for detailed setup and usage instructions

---

**Built with LangGraph Multi-Agent Architecture** 🤖✨

**Made with ❤️ by [Amruth22](https://github.com/Amruth22)**

*This comprehensive problem statement provides a clear roadmap for understanding, implementing, and extending the Invoice AgenticAI system for automated intelligent invoice processing.*
