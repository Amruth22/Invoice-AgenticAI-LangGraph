"""
Document Agent for Invoice Processing
Handles PDF extraction, text processing, and AI-powered invoice parsing
"""

import os
import json
import fitz  # PyMuPDF
import pdfplumber
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from dotenv import load_dotenv

from agents.base_agent import BaseAgent
from state import (
    InvoiceProcessingState, InvoiceData, ItemDetail, 
    ProcessingStatus, ValidationStatus
)
from utils.logger import StructuredLogger

load_dotenv()


class DocumentAgent(BaseAgent):
    """
    Agent responsible for document processing and invoice data extraction
    Uses multiple extraction methods and AI parsing for robust data extraction
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("document", config)
        self.structured_logger = StructuredLogger("document_agent")
        
        # Initialize Gemini AI
        api_key = os.getenv("GEMINI_API_KEY_1")
        if not api_key:
            raise ValueError("GEMINI_API_KEY_1 not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Configuration
        self.extraction_methods = config.get("extraction_methods", ["pymupdf", "pdfplumber"])
        self.ai_confidence_threshold = config.get("ai_confidence_threshold", 0.7)
        self.max_text_length = config.get("max_text_length", 10000)
    
    def _validate_preconditions(self, state: InvoiceProcessingState) -> bool:
        """Validate that we have a file to process"""
        return bool(state.file_name and os.path.exists(f"data/invoices/{state.file_name}"))
    
    def _validate_postconditions(self, state: InvoiceProcessingState) -> bool:
        """Validate that we extracted invoice data successfully"""
        return (
            state.invoice_data is not None and
            state.invoice_data.invoice_number and
            state.invoice_data.customer_name and
            len(state.invoice_data.item_details) > 0
        )
    
    async def execute(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """
        Execute document processing workflow
        """
        try:
            # Step 1: Extract raw text from PDF
            raw_text = await self._extract_text_from_pdf(state.file_name)
            state.raw_text = raw_text
            
            if not raw_text or len(raw_text.strip()) < 50:
                raise ValueError("Insufficient text extracted from PDF")
            
            # Step 2: Parse invoice data using AI
            invoice_data = await self._parse_invoice_with_ai(raw_text)
            
            # Step 3: Validate and enhance extracted data
            enhanced_data = await self._enhance_invoice_data(invoice_data, raw_text)
            
            # Step 4: Calculate confidence score
            confidence = self._calculate_extraction_confidence(enhanced_data, raw_text)
            enhanced_data.extraction_confidence = confidence
            enhanced_data.raw_text = raw_text
            
            # Update state
            state.invoice_data = enhanced_data
            
            # Log successful extraction
            self.structured_logger.log_decision(
                agent_name=self.agent_name,
                process_id=state.process_id,
                decision="extraction_successful",
                reasoning=f"Extracted invoice {enhanced_data.invoice_number} with {confidence:.2f} confidence",
                confidence=confidence
            )
            
            # Determine next workflow step
            if confidence < self.ai_confidence_threshold:
                state.human_review_required = True
                state.human_review_notes = f"Low extraction confidence: {confidence:.2f}"
            
            return state
            
        except Exception as e:
            state.extraction_errors.append(str(e))
            self.structured_logger.log_agent_error(
                agent_name=self.agent_name,
                process_id=state.process_id,
                error=e
            )
            raise
    
    async def _extract_text_from_pdf(self, file_name: str) -> str:
        """
        Extract text from PDF using multiple methods for robustness
        """
        file_path = f"data/invoices/{file_name}"
        extracted_texts = []
        
        # Method 1: PyMuPDF
        if "pymupdf" in self.extraction_methods:
            try:
                text = ""
                with fitz.open(file_path) as doc:
                    for page in doc:
                        text += page.get_text()
                if text.strip():
                    extracted_texts.append(("pymupdf", text))
            except Exception as e:
                self.logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # Method 2: PDFPlumber
        if "pdfplumber" in self.extraction_methods:
            try:
                text = ""
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                if text.strip():
                    extracted_texts.append(("pdfplumber", text))
            except Exception as e:
                self.logger.warning(f"PDFPlumber extraction failed: {e}")
        
        if not extracted_texts:
            raise ValueError("No text could be extracted from PDF")
        
        # Choose the best extraction (longest text usually means better extraction)
        best_method, best_text = max(extracted_texts, key=lambda x: len(x[1]))
        
        self.logger.info(f"Best extraction method: {best_method} ({len(best_text)} characters)")
        
        # Truncate if too long
        if len(best_text) > self.max_text_length:
            best_text = best_text[:self.max_text_length] + "..."
        
        return best_text
    
    async def _parse_invoice_with_ai(self, text: str) -> InvoiceData:
        """
        Parse invoice text using Gemini AI
        """
        prompt = f"""
You are an expert invoice parser. Extract the following fields from the invoice text in pure JSON format.

The invoice contains exactly one or more items. Extract all relevant information accurately.

Return ONLY valid JSON in this exact format:
{{
  "invoice_number": "...",
  "order_id": "...",
  "customer_name": "...",
  "due_date": "...",
  "ship_to": "...",
  "ship_mode": "...",
  "subtotal": 0.0,
  "discount": 0.0,
  "shipping_cost": 0.0,
  "total": 0.0,
  "item_details": [
    {{
      "item_name": "...",
      "quantity": 0,
      "rate": 0.0,
      "amount": 0.0
    }}
  ]
}}

Rules:
1. Extract ALL items, not just the first one
2. Ensure numeric values are properly formatted
3. Use null for missing string fields, 0.0 for missing numeric fields
4. Be precise with item names and descriptions
5. Calculate totals accurately

Invoice text:
\"\"\"{text}\"\"\"
"""
        
        try:
            response = self.model.generate_content(prompt)
            content = response.text.strip()
            
            # Clean up markdown formatting
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            # Parse JSON
            parsed_data = json.loads(content)
            
            # Convert to InvoiceData model
            item_details = [
                ItemDetail(**item) for item in parsed_data.get("item_details", [])
            ]
            
            invoice_data = InvoiceData(
                invoice_number=parsed_data.get("invoice_number", ""),
                order_id=parsed_data.get("order_id", ""),
                customer_name=parsed_data.get("customer_name", ""),
                due_date=parsed_data.get("due_date"),
                ship_to=parsed_data.get("ship_to"),
                ship_mode=parsed_data.get("ship_mode"),
                subtotal=float(parsed_data.get("subtotal", 0.0)),
                discount=float(parsed_data.get("discount", 0.0)),
                shipping_cost=float(parsed_data.get("shipping_cost", 0.0)),
                total=float(parsed_data.get("total", 0.0)),
                item_details=item_details
            )
            
            return invoice_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            self.logger.error(f"Raw AI response: {content}")
            raise ValueError(f"AI returned invalid JSON: {e}")
        except Exception as e:
            self.logger.error(f"AI parsing error: {e}")
            raise ValueError(f"AI parsing failed: {e}")
    
    async def _enhance_invoice_data(self, invoice_data: InvoiceData, raw_text: str) -> InvoiceData:
        """
        Enhance and validate extracted invoice data
        """
        # Validate required fields
        if not invoice_data.invoice_number:
            # Try to extract invoice number using regex patterns
            import re
            patterns = [
                r"Invoice\s*#?\s*:?\s*(\w+)",
                r"Invoice\s*Number\s*:?\s*(\w+)",
                r"INV\s*#?\s*:?\s*(\w+)"
            ]
            for pattern in patterns:
                match = re.search(pattern, raw_text, re.IGNORECASE)
                if match:
                    invoice_data.invoice_number = match.group(1)
                    break
        
        # Validate customer name
        if not invoice_data.customer_name:
            # Try to extract customer name
            patterns = [
                r"Bill\s*To\s*:?\s*([^\n]+)",
                r"Customer\s*:?\s*([^\n]+)",
                r"Sold\s*To\s*:?\s*([^\n]+)"
            ]
            for pattern in patterns:
                match = re.search(pattern, raw_text, re.IGNORECASE)
                if match:
                    invoice_data.customer_name = match.group(1).strip()
                    break
        
        # Validate totals
        if invoice_data.total == 0.0 and invoice_data.item_details:
            # Calculate total from items
            calculated_total = sum(item.amount for item in invoice_data.item_details)
            if calculated_total > 0:
                invoice_data.total = calculated_total
        
        # Add item categories based on item names
        for item in invoice_data.item_details:
            item.category = self._categorize_item(item.item_name)
        
        return invoice_data
    
    def _categorize_item(self, item_name: str) -> str:
        """
        Categorize items based on their names
        """
        item_lower = item_name.lower()
        
        if any(word in item_lower for word in ["chair", "table", "desk", "furniture"]):
            return "Furniture"
        elif any(word in item_lower for word in ["computer", "printer", "phone", "technology"]):
            return "Technology"
        elif any(word in item_lower for word in ["paper", "pen", "supplies", "office"]):
            return "Office Supplies"
        elif any(word in item_lower for word in ["appliance", "kitchen", "toaster"]):
            return "Appliances"
        else:
            return "Other"
    
    def _calculate_extraction_confidence(self, invoice_data: InvoiceData, raw_text: str) -> float:
        """
        Calculate confidence score for extracted data
        """
        confidence_factors = []
        
        # Required field completeness
        required_fields = [
            invoice_data.invoice_number,
            invoice_data.customer_name,
            invoice_data.order_id
        ]
        completeness = sum(1 for field in required_fields if field) / len(required_fields)
        confidence_factors.append(completeness * 0.4)
        
        # Item details quality
        if invoice_data.item_details:
            item_quality = sum(
                1 for item in invoice_data.item_details 
                if item.item_name and item.quantity > 0 and item.rate > 0
            ) / len(invoice_data.item_details)
            confidence_factors.append(item_quality * 0.3)
        else:
            confidence_factors.append(0.0)
        
        # Numeric consistency
        if invoice_data.total > 0 and invoice_data.item_details:
            calculated_total = sum(item.amount for item in invoice_data.item_details)
            if calculated_total > 0:
                total_accuracy = min(invoice_data.total, calculated_total) / max(invoice_data.total, calculated_total)
                confidence_factors.append(total_accuracy * 0.2)
            else:
                confidence_factors.append(0.0)
        else:
            confidence_factors.append(0.0)
        
        # Text quality indicator
        text_quality = min(len(raw_text) / 1000, 1.0)  # Longer text usually means better extraction
        confidence_factors.append(text_quality * 0.1)
        
        return sum(confidence_factors)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for document agent"""
        health_status = await super().health_check()
        
        try:
            # Test AI connection
            test_response = self.model.generate_content("Test connection")
            ai_status = "healthy" if test_response else "unhealthy"
        except Exception as e:
            ai_status = f"unhealthy: {str(e)}"
        
        health_status.update({
            "ai_model_status": ai_status,
            "extraction_methods": self.extraction_methods,
            "confidence_threshold": self.ai_confidence_threshold
        })
        
        return health_status