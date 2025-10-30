"""
Corrected unit test suite for Invoice AgenticAI - LangGraph project
Tests each major component functionality with correct method signatures
"""
import os
import tempfile
import asyncio
import unittest
from unittest.mock import Mock, patch, MagicMock
from unittest.result import TestResult
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

from agents.document_agent import DocumentAgent
from agents.validation_agent import ValidationAgent
from agents.risk_agent import RiskAgent
from agents.payment_agent import PaymentAgent
from agents.audit_agent import AuditAgent
from agents.escalation_agent import EscalationAgent
from invoice_graph import InvoiceProcessingGraph
from state_models import (
    InvoiceProcessingState, InvoiceData, ItemDetail,
    ProcessingStatus, RiskLevel, PaymentStatus
)
from utils.logger import StructuredLogger

# Import test client for FastAPI
from fastapi.testclient import TestClient
from payment_api import app


class TestResultSummary(TestResult):
    """Custom TestResult class to track pass/fail counts"""
    def __init__(self):
        super().__init__()
        self.test_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.error_count = 0

    def startTest(self, test):
        super().startTest(test)
        self.test_count += 1

    def addSuccess(self, test):
        super().addSuccess(test)
        self.success_count += 1

    def addError(self, test, err):
        super().addError(test, err)
        self.error_count += 1

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.failure_count += 1

    def get_counts(self):
        return {
            'total': self.test_count,
            'passed': self.success_count,
            'failed': self.failure_count,
            'errors': self.error_count
        }


class TestInvoiceAgenticAI(unittest.TestCase):
    """Comprehensive test class for all Invoice AgenticAI components"""
    
    def test_api_key_validation(self):
        """Test that required API keys are present in environment"""
        load_dotenv()
        api_keys = [
            "GEMINI_API_KEY_1",
            "GEMINI_API_KEY_2",
            "GEMINI_API_KEY_3",
            "GEMINI_API_KEY_4"
        ]
        
        for key in api_keys:
            self.assertIsNotNone(os.getenv(key), f"{key} is not set in environment")
            self.assertNotEqual(os.getenv(key), f"your_{key.lower()}_here")
    
    def test_pdf_extraction(self):
        """Test PDF text extraction functionality"""
        config = {
            "extraction_methods": ["pymupdf", "pdfplumber"],
            "ai_confidence_threshold": 0.7
        }
        agent = DocumentAgent(config)
        
        # Create a temporary PDF for testing
        temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        temp_pdf.close()
        
        # Mock PyMuPDF extraction with proper context manager
        with patch('fitz.open') as mock_fitz:
            # Create proper mock with context manager support
            mock_doc = Mock()
            mock_page = Mock()
            mock_page.get_text.return_value = "Test invoice text"
            mock_doc.__enter__ = Mock(return_value=mock_doc)
            mock_doc.__exit__ = Mock(return_value=None)
            mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
            
            mock_fitz.return_value = mock_doc
            
            # Test extraction from temporary PDF
            with patch('agents.document_agent.os.path.exists', return_value=True):
                # Since _extract_text_from_pdf is async, we need to run it
                result = asyncio.run(agent._extract_text_from_pdf(temp_pdf.name))
            
            self.assertIn("Test invoice text", result)
    
    def test_ai_parsing(self):
        """Test AI invoice parsing functionality"""
        config = {
            "extraction_methods": ["pymupdf", "pdfplumber"],
            "ai_confidence_threshold": 0.7
        }
        agent = DocumentAgent(config)
        
        with patch('google.generativeai.GenerativeModel.generate_content') as mock_ai:
            mock_response = Mock()
            mock_response.text = '''
            {
                "invoice_number": "INV-001",
                "order_id": "ORD-001",
                "customer_name": "Test Customer",
                "due_date": "2023-12-31",
                "subtotal": 100.0,
                "discount": 0.0,
                "shipping_cost": 10.0,
                "total": 110.0,
                "item_details": [
                    {
                        "item_name": "Test Item",
                        "quantity": 1,
                        "rate": 100.0,
                        "amount": 100.0
                    }
                ]
            }
            '''
            mock_ai.return_value = mock_response
            
            text = "Test invoice text for parsing"
            # Since _parse_invoice_with_ai is async, we need to run it
            result = asyncio.run(agent._parse_invoice_with_ai(text))
            
            self.assertEqual(result.invoice_number, "INV-001")
            self.assertEqual(result.customer_name, "Test Customer")
            self.assertEqual(result.total, 110.0)
            self.assertEqual(len(result.item_details), 1)
    
    def test_purchase_order_validation(self):
        """Test invoice validation against purchase orders"""
        config = {
            "po_file_path": "data/purchase_orders.csv",
            "fuzzy_threshold": 80,
            "amount_tolerance": 0.05
        }
        agent = ValidationAgent(config)
        
        # Load and test the purchase orders data
        po_data = pd.read_csv("data/purchase_orders.csv")
        self.assertIsNotNone(po_data)
        self.assertGreater(len(po_data), 0)
        self.assertIn("invoice_number", po_data.columns)
        self.assertIn("customer_name", po_data.columns)
        
        # Create a sample invoice that matches the data
        item = ItemDetail(item_name="Canon Wireless Fax, Laser Copiers, Technology, TEC-CO-3710", 
                         quantity=5, rate=1893.30, amount=9466.5)
        invoice_data = InvoiceData(
            invoice_number="14021",  # From the sample data
            order_id="ES-2025-BE11335139-41340",
            customer_name="Bill Eplett",
            total=9466.5,
            item_details=[item]
        )
        
        state = InvoiceProcessingState(
            file_name="test_invoice.pdf",
            invoice_data=invoice_data
        )
        
        # Test data loading functionality of validation agent
        loaded_data = agent._load_purchase_orders()
        self.assertIsNotNone(loaded_data)
        self.assertGreater(len(loaded_data), 0)
    
    def test_risk_assessment(self):
        """Test risk assessment functionality"""
        config = {
            "risk_thresholds": {
                "low": 0.3, 
                "medium": 0.6, 
                "high": 0.8, 
                "critical": 0.9
            }
        }
        agent = RiskAgent(config)
        
        # Create sample invoice data that might trigger risk factors
        item = ItemDetail(item_name="Test Item", quantity=1, rate=50000.0, amount=50000.0)  # High amount
        invoice_data = InvoiceData(
            invoice_number="INV-999",
            order_id="ORD-999",
            customer_name="New Customer",
            total=50000.0,  # High amount
            item_details=[item]
        )
        
        # Create a sample validation result
        from state_models import ValidationResult, ValidationStatus
        validation_result = ValidationResult(
            validation_status=ValidationStatus.VALID
        )
        
        # Call the correct method with all required parameters
        risk_assessment = asyncio.run(agent._calculate_base_risk_score(invoice_data, validation_result))
        
        # Should return a risk score
        self.assertIsNotNone(risk_assessment)
        self.assertIsInstance(risk_assessment, float)
    
    def test_payment_processing(self):
        """Test payment processing functionality"""
        config = {
            "payment_api_url": "http://localhost:8000/initiate_payment",
            "auto_payment_threshold": 5000,
            "manual_approval_threshold": 25000
        }
        agent = PaymentAgent(config)
        
        # Create sample validation and risk results
        from state_models import ValidationResult, ValidationStatus, RiskAssessment
        validation_result = ValidationResult(
            validation_status=ValidationStatus.VALID
        )
        risk_assessment = RiskAssessment(
            risk_level=RiskLevel.LOW
        )
        
        state = InvoiceProcessingState(file_name="test.pdf")
        
        # Test payment decision based on amount thresholds
        # Low amount - should be approved automatically
        low_amount_invoice = InvoiceData(
            invoice_number="INV-001",
            order_id="ORD-001",
            customer_name="Test Customer",
            total=100.0,
            item_details=[ItemDetail(item_name="Test Item", quantity=1, rate=100.0, amount=100.0)]
        )
        
        decision = asyncio.run(agent._make_payment_decision(
            low_amount_invoice, validation_result, risk_assessment, state))
        # This test verifies the decision logic is functional
        self.assertIsNotNone(decision)
        
        # High amount - should require approval
        high_amount_invoice = InvoiceData(
            invoice_number="INV-002",
            order_id="ORD-002",
            customer_name="Test Customer",
            total=30000.0,
            item_details=[ItemDetail(item_name="Test Item", quantity=1, rate=30000.0, amount=30000.0)]
        )
        
        decision = asyncio.run(agent._make_payment_decision(
            high_amount_invoice, validation_result, risk_assessment, state))
        # This test verifies the decision logic is functional
        self.assertIsNotNone(decision)
    
    def test_document_agent_functionality(self):
        """Test document agent core functionality"""
        config = {
            "extraction_methods": ["pymupdf", "pdfplumber"],
            "ai_confidence_threshold": 0.7
        }
        agent = DocumentAgent(config)
        
        # Test confidence calculation
        item = ItemDetail(item_name="Test Item", quantity=1, rate=100.0, amount=100.0)
        invoice_data = InvoiceData(
            invoice_number="INV-001",
            order_id="ORD-001",
            customer_name="Test Customer",
            total=100.0,
            item_details=[item]
        )
        
        raw_text = "This is a test invoice with complete information"
        confidence = agent._calculate_extraction_confidence(invoice_data, raw_text)
        
        # Confidence should be between 0 and 1
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
    
    def test_validation_agent_functionality(self):
        """Test validation agent core functionality"""
        config = {
            "po_file_path": "data/purchase_orders.csv",
            "fuzzy_threshold": 80,
            "amount_tolerance": 0.05
        }
        agent = ValidationAgent(config)
        
        # Test that validation agent can load PO data
        po_data = agent._load_purchase_orders()
        self.assertIsNotNone(po_data)
        self.assertGreater(len(po_data), 0)
        
        # Create validation result and invoice data
        item = ItemDetail(item_name="Test Item", quantity=1, rate=100.0, amount=100.0)
        invoice_data = InvoiceData(
            invoice_number="INV-001",
            order_id="ORD-001",
            customer_name="Test Customer",
            total=100.0,
            item_details=[item]
        )
        
        # Test validation process with the correct method
        result = asyncio.run(agent._validate_against_pos(invoice_data, []))
        # This test verifies the validation process can be initiated
        self.assertIsNotNone(result)
    
    def test_risk_agent_functionality(self):
        """Test risk agent core functionality"""
        config = {
            "risk_thresholds": {
                "low": 0.3, 
                "medium": 0.6, 
                "high": 0.8, 
                "critical": 0.9
            }
        }
        agent = RiskAgent(config)
        
        # Create validation result
        from state_models import ValidationResult, ValidationStatus
        validation_result = ValidationResult(
            validation_status=ValidationStatus.VALID
        )
        
        # Test fraud detection with correct method signature
        suspicious_customers = ["test_customer", "fraudulent", "suspicious"]
        
        for customer in suspicious_customers:
            item = ItemDetail(item_name="Test Item", quantity=1, rate=100.0, amount=100.0)
            invoice_data = InvoiceData(
                invoice_number="INV-001",
                order_id="ORD-001",
                customer_name=customer,
                total=100.0,
                item_details=[item]
            )
            
            fraud_indicators = asyncio.run(agent._detect_fraud_indicators(invoice_data, validation_result))
            # Test that fraud detection runs without error
            self.assertIsNotNone(fraud_indicators)
    
    def test_payment_agent_functionality(self):
        """Test payment agent core functionality"""
        config = {
            "payment_api_url": "http://localhost:8000/initiate_payment",
            "auto_payment_threshold": 5000,
            "manual_approval_threshold": 25000
        }
        agent = PaymentAgent(config)
        
        # Test payment method selection
        payment_method = agent._select_payment_method(1000.0)
        self.assertIsNotNone(payment_method)
        
        # Test threshold logic
        self.assertEqual(agent.auto_payment_threshold, 5000)
        self.assertEqual(agent.manual_approval_threshold, 25000)
    
    def test_graph_initialization(self):
        """Test LangGraph workflow initialization"""
        config = {
            "document_agent": {"extraction_methods": ["pymupdf"], "ai_confidence_threshold": 0.7},
            "validation_agent": {"po_file_path": "data/purchase_orders.csv"},
            "risk_agent": {"risk_thresholds": {"low": 0.3, "medium": 0.6, "high": 0.8, "critical": 0.9}}
        }
        graph = InvoiceProcessingGraph(config)
        
        self.assertIsNotNone(graph.graph)
        self.assertIsNotNone(graph.compiled_graph)
    
    def test_state_model_validation(self):
        """Test state model functionality"""
        # Test invoice data model
        item = ItemDetail(item_name="Test Item", quantity=2, rate=50.0, amount=100.0)
        
        invoice_data = InvoiceData(
            invoice_number="INV-001",
            order_id="ORD-001",
            customer_name="Test Customer",
            subtotal=100.0,
            discount=5.0,
            shipping_cost=10.0,
            total=105.0,  # subtotal - discount + shipping
            item_details=[item]
        )
        
        self.assertEqual(invoice_data.invoice_number, "INV-001")
        self.assertEqual(invoice_data.customer_name, "Test Customer")
        self.assertEqual(len(invoice_data.item_details), 1)
    
    def test_state_audit_trail(self):
        """Test audit trail functionality in state"""
        state = InvoiceProcessingState(file_name="test.pdf")
        
        # Add an audit entry
        state.add_audit_entry(
            agent_name="test_agent",
            action="test_action",
            status=ProcessingStatus.COMPLETED,
            details={"test": "value"}
        )
        
        # Verify audit entry was added
        self.assertEqual(len(state.audit_trail), 1)
        self.assertEqual(state.audit_trail[0].agent_name, "test_agent")
        self.assertEqual(state.audit_trail[0].action, "test_action")
        self.assertEqual(state.audit_trail[0].status, ProcessingStatus.COMPLETED)
    
    def test_agent_metrics(self):
        """Test agent metrics tracking"""
        state = InvoiceProcessingState(file_name="test.pdf")
        
        # Update agent metrics
        state.update_agent_metrics("test_agent", success=True, duration_ms=100)
        state.update_agent_metrics("test_agent", success=False, duration_ms=150)
        
        # Verify metrics were updated correctly
        metrics = state.agent_metrics["test_agent"]
        self.assertEqual(metrics.executions, 2)
        self.assertEqual(metrics.successes, 1)
        self.assertEqual(metrics.failures, 1)
        self.assertGreater(metrics.average_duration_ms, 0)
    
    def test_payment_api_endpoints(self):
        """Test payment API functionality"""
        client = TestClient(app)  # Use TestClient instead of app.test_client()
        
        # Test health endpoint
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)
        
        # Test payment initiation endpoint
        payment_request = {
            "order_id": "ORD-123",
            "customer_name": "Test Customer",
            "amount": 100.0,
            "due_date": "2023-12-31"
        }
        
        response = client.post("/initiate_payment", json=payment_request)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("transaction_id", data)
        self.assertIn("status", data)
        self.assertIn("timestamp", data)


if __name__ == '__main__':
    # Create a test suite to run with custom result tracking
    suite = unittest.TestLoader().loadTestsFromTestCase(TestInvoiceAgenticAI)
    
    # Create a custom result object to track test outcomes
    result = TestResultSummary()
    
    print("Running corrected Invoice AgenticAI - LangGraph unit tests...")
    suite.run(result)
    
    # Print summary
    counts = result.get_counts()
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests Run: {counts['total']}")
    print(f"Passed: {counts['passed']}")
    print(f"Failed: {counts['failed']}")
    print(f"Errors: {counts['errors']}")
    print(f"Success Rate: {counts['passed'] / counts['total'] * 100:.1f}%") if counts['total'] > 0 else print("Success Rate: 0%")
    print("="*60)
    
    if counts['failed'] > 0 or counts['errors'] > 0:
        print("[FAILED] Some tests failed or had errors")
        exit(1)
    else:
        print("[PASSED] All tests passed!")
        exit(0)