"""
Validation Agent for Invoice Processing
Handles purchase order matching, discrepancy detection, and validation scoring
"""

import pandas as pd
from typing import Dict, Any, List, Tuple
from fuzzywuzzy import fuzz
import numpy as np

from agents.base_agent import BaseAgent
from state import (
    InvoiceProcessingState, ValidationResult, ValidationStatus,
    ProcessingStatus
)
from utils.logger import StructuredLogger


class ValidationAgent(BaseAgent):
    """
    Agent responsible for validating invoice data against purchase orders
    Performs fuzzy matching, discrepancy detection, and validation scoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("validation", config)
        self.structured_logger = StructuredLogger("validation_agent")
        
        # Configuration
        self.po_file_path = config.get("po_file_path", "data/purchase_orders.csv")
        self.fuzzy_threshold = config.get("fuzzy_threshold", 80)
        self.amount_tolerance = config.get("amount_tolerance", 0.05)  # 5% tolerance
        self.rate_tolerance = config.get("rate_tolerance", 0.02)  # 2% tolerance
        
        # Load purchase orders
        self.po_df = self._load_purchase_orders()
    
    def _validate_preconditions(self, state: InvoiceProcessingState) -> bool:
        """Validate that we have invoice data to validate"""
        return (
            state.invoice_data is not None and
            state.invoice_data.invoice_number and
            state.invoice_data.customer_name and
            len(state.invoice_data.item_details) > 0
        )
    
    def _validate_postconditions(self, state: InvoiceProcessingState) -> bool:
        """Validate that validation was completed"""
        return (
            state.validation_result is not None and
            state.validation_result.validation_status is not None
        )
    
    async def execute(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """
        Execute validation workflow
        """
        try:
            invoice_data = state.invoice_data
            
            # Step 1: Find matching purchase orders
            matching_pos = await self._find_matching_pos(invoice_data)
            
            # Step 2: Perform detailed validation
            validation_result = await self._validate_against_pos(invoice_data, matching_pos)
            
            # Step 3: Calculate confidence score
            confidence = self._calculate_validation_confidence(validation_result, matching_pos)
            validation_result.confidence_score = confidence
            
            # Step 4: Determine validation status
            validation_result.validation_status = self._determine_validation_status(validation_result)
            
            # Update state
            state.validation_result = validation_result
            
            # Log validation decision
            self.structured_logger.log_decision(
                agent_name=self.agent_name,
                process_id=state.process_id,
                decision=validation_result.validation_status.value,
                reasoning=validation_result.validation_result,
                confidence=confidence
            )
            
            # Determine if escalation is needed
            if validation_result.validation_status in [ValidationStatus.REQUIRES_APPROVAL, ValidationStatus.INVALID]:
                if self._should_escalate_validation(validation_result, invoice_data):
                    state.escalation_required = True
                    state.escalation_reason = f"Validation issues: {validation_result.validation_result}"
            
            return state
            
        except Exception as e:
            state.validation_errors.append(str(e))
            self.structured_logger.log_agent_error(
                agent_name=self.agent_name,
                process_id=state.process_id,
                error=e
            )
            raise
    
    def _load_purchase_orders(self) -> pd.DataFrame:
        """Load and preprocess purchase orders"""
        try:
            po_df = pd.read_csv(self.po_file_path)
            
            # Ensure required columns exist
            required_columns = [
                "invoice_number", "order_id", "customer_name", 
                "quantity", "rate", "expected_amount"
            ]
            
            missing_columns = [col for col in required_columns if col not in po_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in PO file: {missing_columns}")
            
            # Convert data types
            po_df["invoice_number"] = po_df["invoice_number"].astype(str)
            po_df["order_id"] = po_df["order_id"].astype(str)
            po_df["customer_name"] = po_df["customer_name"].astype(str)
            po_df["quantity"] = pd.to_numeric(po_df["quantity"], errors="coerce")
            po_df["rate"] = pd.to_numeric(po_df["rate"], errors="coerce")
            po_df["expected_amount"] = pd.to_numeric(po_df["expected_amount"], errors="coerce")
            
            self.logger.info(f"Loaded {len(po_df)} purchase orders")
            return po_df
            
        except Exception as e:
            self.logger.error(f"Failed to load purchase orders: {e}")
            raise
    
    async def _find_matching_pos(self, invoice_data) -> List[Dict[str, Any]]:
        """
        Find matching purchase orders using multiple matching strategies
        """
        matching_pos = []
        
        # Strategy 1: Exact match on invoice_number, order_id, customer_name
        exact_matches = self.po_df[
            (self.po_df["invoice_number"] == invoice_data.invoice_number) &
            (self.po_df["order_id"] == invoice_data.order_id) &
            (self.po_df["customer_name"] == invoice_data.customer_name)
        ]
        
        if not exact_matches.empty:
            for _, po_row in exact_matches.iterrows():
                matching_pos.append({
                    "match_type": "exact",
                    "match_score": 100,
                    "po_data": po_row.to_dict()
                })
        
        # Strategy 2: Fuzzy match on customer name + exact order_id
        if not matching_pos:
            for _, po_row in self.po_df.iterrows():
                if po_row["order_id"] == invoice_data.order_id:
                    customer_similarity = fuzz.ratio(
                        invoice_data.customer_name.lower(),
                        po_row["customer_name"].lower()
                    )
                    
                    if customer_similarity >= self.fuzzy_threshold:
                        matching_pos.append({
                            "match_type": "fuzzy_customer",
                            "match_score": customer_similarity,
                            "po_data": po_row.to_dict()
                        })
        
        # Strategy 3: Partial match on invoice_number
        if not matching_pos:
            for _, po_row in self.po_df.iterrows():
                if po_row["invoice_number"] == invoice_data.invoice_number:
                    matching_pos.append({
                        "match_type": "invoice_number",
                        "match_score": 90,
                        "po_data": po_row.to_dict()
                    })
        
        # Sort by match score
        matching_pos.sort(key=lambda x: x["match_score"], reverse=True)
        
        self.logger.info(f"Found {len(matching_pos)} matching POs for invoice {invoice_data.invoice_number}")
        return matching_pos
    
    async def _validate_against_pos(self, invoice_data, matching_pos: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate invoice data against matching purchase orders
        """
        validation_result = ValidationResult()
        
        if not matching_pos:
            validation_result.po_found = False
            validation_result.validation_result = "No matching purchase order found"
            validation_result.discrepancies = ["Missing PO"]
            return validation_result
        
        # Use the best matching PO
        best_match = matching_pos[0]
        po_data = best_match["po_data"]
        validation_result.po_found = True
        validation_result.po_data = po_data
        
        # Validate each invoice item against PO
        discrepancies = []
        
        for item in invoice_data.item_details:
            item_discrepancies = self._validate_item_against_po(item, po_data)
            discrepancies.extend(item_discrepancies)
        
        # Validate totals
        total_discrepancies = self._validate_totals(invoice_data, po_data)
        discrepancies.extend(total_discrepancies)
        
        # Set validation flags
        validation_result.quantity_match = not any("Quantity" in d for d in discrepancies)
        validation_result.rate_match = not any("Rate" in d for d in discrepancies)
        validation_result.amount_match = not any("Amount" in d for d in discrepancies)
        validation_result.discrepancies = discrepancies
        
        # Generate validation result summary
        if not discrepancies:
            validation_result.validation_result = "Perfect match"
        else:
            validation_result.validation_result = "; ".join(discrepancies)
        
        return validation_result
    
    def _validate_item_against_po(self, item, po_data: Dict[str, Any]) -> List[str]:
        """
        Validate individual item against purchase order
        """
        discrepancies = []
        
        # Quantity validation
        po_quantity = po_data.get("quantity", 0)
        if item.quantity != po_quantity:
            discrepancy_pct = abs(item.quantity - po_quantity) / max(po_quantity, 1) * 100
            discrepancies.append(f"Quantity mismatch: Invoice={item.quantity}, PO={po_quantity} ({discrepancy_pct:.1f}% diff)")
        
        # Rate validation
        po_rate = po_data.get("rate", 0.0)
        if po_rate > 0:
            rate_diff = abs(item.rate - po_rate) / po_rate
            if rate_diff > self.rate_tolerance:
                discrepancy_pct = rate_diff * 100
                discrepancies.append(f"Rate mismatch: Invoice=${item.rate:.2f}, PO=${po_rate:.2f} ({discrepancy_pct:.1f}% diff)")
        
        # Amount validation
        expected_amount = po_data.get("expected_amount", 0.0)
        if expected_amount > 0:
            amount_diff = abs(item.amount - expected_amount) / expected_amount
            if amount_diff > self.amount_tolerance:
                discrepancy_pct = amount_diff * 100
                if item.amount > expected_amount:
                    discrepancies.append(f"Overbilling: Invoice=${item.amount:.2f}, Expected=${expected_amount:.2f} ({discrepancy_pct:.1f}% over)")
                else:
                    discrepancies.append(f"Underbilling: Invoice=${item.amount:.2f}, Expected=${expected_amount:.2f} ({discrepancy_pct:.1f}% under)")
        
        return discrepancies
    
    def _validate_totals(self, invoice_data, po_data: Dict[str, Any]) -> List[str]:
        """
        Validate invoice totals against expected amounts
        """
        discrepancies = []
        
        expected_total = po_data.get("expected_amount", 0.0)
        if expected_total > 0 and invoice_data.total > 0:
            total_diff = abs(invoice_data.total - expected_total) / expected_total
            if total_diff > self.amount_tolerance:
                discrepancy_pct = total_diff * 100
                if invoice_data.total > expected_total:
                    discrepancies.append(f"Total overbilling: ${invoice_data.total:.2f} vs ${expected_total:.2f} ({discrepancy_pct:.1f}% over)")
                else:
                    discrepancies.append(f"Total underbilling: ${invoice_data.total:.2f} vs ${expected_total:.2f} ({discrepancy_pct:.1f}% under)")
        
        return discrepancies
    
    def _calculate_validation_confidence(self, validation_result: ValidationResult, 
                                       matching_pos: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for validation results
        """
        confidence_factors = []
        
        # PO match quality
        if matching_pos:
            best_match_score = matching_pos[0]["match_score"] / 100.0
            confidence_factors.append(best_match_score * 0.3)
        else:
            confidence_factors.append(0.0)
        
        # Validation completeness
        validation_checks = [
            validation_result.po_found,
            validation_result.quantity_match,
            validation_result.rate_match,
            validation_result.amount_match
        ]
        completeness = sum(validation_checks) / len(validation_checks)
        confidence_factors.append(completeness * 0.4)
        
        # Discrepancy severity
        if validation_result.discrepancies:
            # Lower confidence for more discrepancies
            severity_penalty = min(len(validation_result.discrepancies) * 0.1, 0.3)
            confidence_factors.append(max(0.0, 0.3 - severity_penalty))
        else:
            confidence_factors.append(0.3)
        
        return sum(confidence_factors)
    
    def _determine_validation_status(self, validation_result: ValidationResult) -> ValidationStatus:
        """
        Determine overall validation status based on results
        """
        if not validation_result.po_found:
            return ValidationStatus.MISSING_PO
        
        if not validation_result.discrepancies:
            return ValidationStatus.VALID
        
        # Check severity of discrepancies
        critical_issues = [
            d for d in validation_result.discrepancies 
            if any(keyword in d.lower() for keyword in ["overbilling", "critical", "fraud"])
        ]
        
        if critical_issues:
            return ValidationStatus.REQUIRES_APPROVAL
        
        # Check if discrepancies are within acceptable tolerance
        minor_issues = [
            d for d in validation_result.discrepancies
            if any(keyword in d.lower() for keyword in ["mismatch", "diff"])
        ]
        
        if len(minor_issues) <= 2 and validation_result.confidence_score > 0.7:
            return ValidationStatus.PARTIAL_MATCH
        
        return ValidationStatus.INVALID
    
    def _should_escalate_validation(self, validation_result: ValidationResult, invoice_data) -> bool:
        """
        Determine if validation issues require escalation
        """
        escalation_conditions = [
            # High-value invoices with discrepancies
            invoice_data.total > 10000 and validation_result.discrepancies,
            
            # Overbilling detected
            any("overbilling" in d.lower() for d in validation_result.discrepancies),
            
            # Multiple critical discrepancies
            len(validation_result.discrepancies) > 3,
            
            # Low confidence with discrepancies
            validation_result.confidence_score < 0.5 and validation_result.discrepancies
        ]
        
        return any(escalation_conditions)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for validation agent"""
        health_status = await super().health_check()
        
        try:
            po_count = len(self.po_df) if self.po_df is not None else 0
            po_status = "healthy" if po_count > 0 else "unhealthy"
        except Exception as e:
            po_status = f"unhealthy: {str(e)}"
            po_count = 0
        
        health_status.update({
            "po_database_status": po_status,
            "po_count": po_count,
            "fuzzy_threshold": self.fuzzy_threshold,
            "amount_tolerance": self.amount_tolerance
        })
        
        return health_status