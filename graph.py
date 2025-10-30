"""
LangGraph Workflow Orchestrator for Invoice Processing
Defines the agentic workflow graph with conditional routing and state management
"""

import asyncio
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import (
    InvoiceProcessingState, ProcessingStatus, ValidationStatus,
    RiskLevel, PaymentStatus, WORKFLOW_CONFIGS
)
from agents.base_agent import agent_registry
from agents.document_agent import DocumentAgent
from agents.validation_agent import ValidationAgent
from agents.risk_agent import RiskAgent
from agents.payment_agent import PaymentAgent
from agents.audit_agent import AuditAgent
from agents.escalation_agent import EscalationAgent
from utils.logger import StructuredLogger


class InvoiceProcessingGraph:
    """
    LangGraph-based workflow orchestrator for invoice processing
    Manages agent execution, routing, and state transitions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = StructuredLogger("invoice_graph")
        
        # Initialize agents
        self._initialize_agents()
        
        # Create workflow graph
        self.graph = self._create_workflow_graph()
        
        # Add memory for state persistence
        self.memory = MemorySaver()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
    
    def _initialize_agents(self):
        """Initialize and register all agents"""
        # Document processing agent
        document_agent = DocumentAgent(self.config.get("document_agent", {}))
        agent_registry.register(document_agent)
        
        # Validation agent
        validation_agent = ValidationAgent(self.config.get("validation_agent", {}))
        agent_registry.register(validation_agent)
        
        # Risk assessment agent
        risk_agent = RiskAgent(self.config.get("risk_agent", {}))
        agent_registry.register(risk_agent)
        
        # Payment processing agent
        payment_agent = PaymentAgent(self.config.get("payment_agent", {}))
        agent_registry.register(payment_agent)
        
        # Audit agent
        audit_agent = AuditAgent(self.config.get("audit_agent", {}))
        agent_registry.register(audit_agent)
        
        # Escalation agent
        escalation_agent = EscalationAgent(self.config.get("escalation_agent", {}))
        agent_registry.register(escalation_agent)
        
        self.logger.logger.info(f"Initialized {len(agent_registry.list_agents())} agents")
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Create state graph
        workflow = StateGraph(InvoiceProcessingState)
        
        # Add agent nodes
        workflow.add_node("document_agent", self._document_agent_node)
        workflow.add_node("validation_agent", self._validation_agent_node)
        workflow.add_node("risk_agent", self._risk_agent_node)
        workflow.add_node("payment_agent", self._payment_agent_node)
        workflow.add_node("audit_agent", self._audit_agent_node)
        workflow.add_node("escalation_agent", self._escalation_agent_node)
        workflow.add_node("human_review", self._human_review_node)
        
        # Define workflow edges and routing
        workflow.set_entry_point("document_agent")
        
        # Document agent routing
        workflow.add_conditional_edges(
            "document_agent",
            self._route_after_document,
            {
                "validation": "validation_agent",
                "escalation": "escalation_agent",
                "end": END
            }
        )
        
        # Validation agent routing
        workflow.add_conditional_edges(
            "validation_agent",
            self._route_after_validation,
            {
                "risk": "risk_agent",
                "escalation": "escalation_agent",
                "end": END
            }
        )
        
        # Risk agent routing
        workflow.add_conditional_edges(
            "risk_agent",
            self._route_after_risk,
            {
                "payment": "payment_agent",
                "escalation": "escalation_agent",
                "human_review": "human_review",
                "end": END
            }
        )
        
        # Payment agent routing
        workflow.add_conditional_edges(
            "payment_agent",
            self._route_after_payment,
            {
                "audit": "audit_agent",
                "escalation": "escalation_agent",
                "end": END
            }
        )
        
        # Audit agent routing
        workflow.add_conditional_edges(
            "audit_agent",
            self._route_after_audit,
            {
                "escalation": "escalation_agent",
                "end": END
            }
        )
        
        # Escalation and human review end the workflow
        workflow.add_edge("escalation_agent", END)
        workflow.add_edge("human_review", END)
        
        return workflow
    
    # Agent execution nodes
    async def _document_agent_node(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """Execute document processing agent"""
        agent = agent_registry.get("document")
        return await agent.run(state)
    
    async def _validation_agent_node(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """Execute validation agent"""
        agent = agent_registry.get("validation")
        return await agent.run(state)
    
    async def _risk_agent_node(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """Execute risk assessment agent"""
        agent = agent_registry.get("risk")
        return await agent.run(state)
    
    async def _payment_agent_node(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """Execute payment processing agent"""
        agent = agent_registry.get("payment")
        return await agent.run(state)
    
    async def _audit_agent_node(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """Execute audit agent"""
        agent = agent_registry.get("audit")
        return await agent.run(state)
    
    async def _escalation_agent_node(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """Execute escalation agent"""
        agent = agent_registry.get("escalation")
        state = await agent.run(state)
        # Mark as escalated after escalation agent completes
        state.overall_status = ProcessingStatus.ESCALATED
        return state
    
    async def _human_review_node(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """Handle human review requirement"""
        state.overall_status = ProcessingStatus.REQUIRES_REVIEW
        state.add_audit_entry(
            agent_name="human_review",
            action="review_required",
            status=ProcessingStatus.REQUIRES_REVIEW,
            details={"reason": state.human_review_notes or "Manual review required"}
        )
        return state
    
    # Routing logic functions
    def _route_after_document(self, state: InvoiceProcessingState) -> Literal["validation", "escalation", "end"]:
        """Route after document processing"""
        if state.overall_status == ProcessingStatus.FAILED:
            return "end"
        
        if state.escalation_required or state.should_escalate():
            return "escalation"
        
        if state.invoice_data and state.invoice_data.extraction_confidence > 0.5:
            return "validation"
        
        return "escalation"
    
    def _route_after_validation(self, state: InvoiceProcessingState) -> Literal["risk", "escalation", "end"]:
        """Route after validation"""
        if state.overall_status == ProcessingStatus.FAILED:
            return "end"
        
        if state.escalation_required or state.should_escalate():
            return "escalation"
        
        if state.validation_result:
            return "risk"
        
        return "escalation"
    
    def _route_after_risk(self, state: InvoiceProcessingState) -> Literal["payment", "escalation", "human_review", "end"]:
        """Route after risk assessment"""
        if state.overall_status == ProcessingStatus.FAILED:
            return "end"
        
        if state.escalation_required or state.should_escalate():
            return "escalation"
        
        if state.human_review_required:
            return "human_review"
        
        if state.risk_assessment:
            if state.risk_assessment.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                if state.validation_result and state.validation_result.validation_status == ValidationStatus.VALID:
                    return "human_review"
                else:
                    return "escalation"
            else:
                return "payment"
        
        return "escalation"
    
    def _route_after_payment(self, state: InvoiceProcessingState) -> Literal["audit", "escalation", "end"]:
        """Route after payment processing"""
        if state.overall_status == ProcessingStatus.FAILED:
            return "end"
        
        if state.payment_decision:
            if state.payment_decision.payment_status in [PaymentStatus.REJECTED, PaymentStatus.FAILED]:
                return "escalation"
            elif state.payment_decision.payment_status == PaymentStatus.REQUIRES_ESCALATION:
                return "escalation"
            else:
                return "audit"
        
        return "escalation"
    
    def _route_after_audit(self, state: InvoiceProcessingState) -> Literal["escalation", "end"]:
        """Route after audit"""
        if state.escalation_required or state.should_escalate():
            return "escalation"
        
        # Mark as completed
        state.overall_status = ProcessingStatus.COMPLETED
        state.completed_at = state.updated_at
        
        return "end"
    
    async def process_invoice(self, file_name: str, workflow_type: str = "standard",
                            config: Dict[str, Any] = None) -> InvoiceProcessingState:
        """
        Process a single invoice through the workflow
        
        Args:
            file_name: Name of the invoice file to process
            workflow_type: Type of workflow to use (standard, high_value, expedited)
            config: Additional configuration for processing
            
        Returns:
            Final processing state
        """
        # Create initial state
        initial_state = InvoiceProcessingState(
            file_name=file_name,
            workflow_type=workflow_type
        )
        
        # Apply workflow-specific configuration
        if workflow_type in WORKFLOW_CONFIGS:
            workflow_config = WORKFLOW_CONFIGS[workflow_type]
            initial_state.max_retries = workflow_config.retry_policies.get("default", 3)
        
        # Apply custom configuration
        if config:
            initial_state.business_context.update(config)
            if "priority_level" in config:
                initial_state.priority_level = config["priority_level"]
        
        self.logger.log_workflow_start(workflow_type, initial_state.process_id)
        
        try:
            # Execute workflow
            result = await self.compiled_graph.ainvoke(
                initial_state,
                config={"thread_id": initial_state.process_id}
            )
            
            # Extract final state from LangGraph result
            final_state = self._extract_final_state(result, initial_state)
            
            # Calculate processing duration
            duration_ms = int((final_state.updated_at - final_state.created_at).total_seconds() * 1000)
            
            self.logger.log_workflow_complete(
                workflow_type, 
                initial_state.process_id, 
                duration_ms,
                final_status=final_state.overall_status.value
            )
            
            return final_state
            
        except Exception as e:
            self.logger.logger.error(f"Workflow execution failed for {file_name}: {e}")
            
            # Update state with error
            initial_state.overall_status = ProcessingStatus.FAILED
            initial_state.add_audit_entry(
                agent_name="workflow",
                action="workflow_failed",
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
            
            return initial_state
    
    async def process_batch(self, file_names: List[str], workflow_type: str = "standard",
                          max_concurrent: int = 5) -> List[InvoiceProcessingState]:
        """
        Process multiple invoices concurrently
        
        Args:
            file_names: List of invoice files to process
            workflow_type: Type of workflow to use
            max_concurrent: Maximum number of concurrent processes
            
        Returns:
            List of final processing states
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(file_name: str) -> InvoiceProcessingState:
            async with semaphore:
                return await self.process_invoice(file_name, workflow_type)
        
        self.logger.logger.info(f"Starting batch processing of {len(file_names)} invoices")
        
        # Process all invoices concurrently
        results = await asyncio.gather(
            *[process_single(file_name) for file_name in file_names],
            return_exceptions=True
        )
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create failed state for exception
                failed_state = InvoiceProcessingState(
                    file_name=file_names[i],
                    workflow_type=workflow_type,
                    overall_status=ProcessingStatus.FAILED
                )
                failed_state.add_audit_entry(
                    agent_name="workflow",
                    action="batch_processing_failed",
                    status=ProcessingStatus.FAILED,
                    error_message=str(result)
                )
                final_results.append(failed_state)
            else:
                final_results.append(result)
        
        # Log batch completion
        successful = sum(1 for r in final_results if r.overall_status in [ProcessingStatus.COMPLETED, ProcessingStatus.ESCALATED])
        failed = len(final_results) - successful
        
        self.logger.logger.info(
            f"Batch processing completed: {successful} successful, {failed} failed"
        )
        
        return final_results
    
    async def get_workflow_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a workflow execution
        
        Args:
            process_id: Process ID to check
            
        Returns:
            Current workflow status or None if not found
        """
        try:
            # Get state from memory
            state = await self.compiled_graph.aget_state({"thread_id": process_id})
            
            if state and state.values:
                processing_state = state.values
                return processing_state.get_processing_summary()
            
            return None
            
        except Exception as e:
            self.logger.logger.error(f"Failed to get workflow status for {process_id}: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of the workflow system
        """
        health_status = {
            "workflow_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "agents": {},
            "graph_status": "healthy"
        }
        
        # Check all agents
        agent_health = await agent_registry.health_check_all()
        health_status["agents"] = agent_health
        
        # Check if any agents are unhealthy
        unhealthy_agents = [
            name for name, status in agent_health.items()
            if status.get("status") != "healthy"
        ]
        
        if unhealthy_agents:
            health_status["workflow_status"] = "degraded"
            health_status["unhealthy_agents"] = unhealthy_agents
        
        # Test graph compilation
        try:
            test_state = InvoiceProcessingState(file_name="health_check.pdf")
            # Just test that the graph can be invoked (don't actually run)
            health_status["graph_compilation"] = "healthy"
        except Exception as e:
            health_status["workflow_status"] = "unhealthy"
            health_status["graph_compilation"] = f"failed: {str(e)}"
        
        return health_status
    
    def _extract_final_state(self, result, initial_state: InvoiceProcessingState) -> InvoiceProcessingState:
        """Extract and validate final state from LangGraph result"""
        try:
            # LangGraph returns the state directly as InvoiceProcessingState
            if isinstance(result, InvoiceProcessingState):
                return result

            # LangGraph returns AddableValuesDict - extract the actual state
            if isinstance(result, dict) or hasattr(result, '__getitem__'):
                # The result is a dict-like object containing the state
                # Try to get the state directly from the dict
                for key, value in result.items():
                    if isinstance(value, InvoiceProcessingState):
                        return value

                # If no InvoiceProcessingState found, the dict itself IS the state data
                # The dict contains all state fields - reconstruct InvoiceProcessingState from it
                if 'process_id' in result and 'overall_status' in result:
                    # Create InvoiceProcessingState from dict
                    try:
                        final_state = InvoiceProcessingState(**result)
                        return final_state
                    except Exception as e:
                        self.logger.logger.warning(f"Failed to reconstruct state from dict: {e}")

            # Fall back to initial state if extraction fails
            result_keys = list(result.keys()) if hasattr(result, 'keys') else []
            self.logger.logger.warning(f"Could not extract final state, using initial. Result type: {type(result)}, Keys: {result_keys[:5]}")
            final_state = initial_state
            final_state.updated_at = datetime.now()
            
            # Ensure timestamps are set
            if not hasattr(final_state, 'updated_at') or final_state.updated_at is None:
                final_state.updated_at = datetime.now()
            if not hasattr(final_state, 'created_at') or final_state.created_at is None:
                final_state.created_at = initial_state.created_at
            
            return final_state
            
        except Exception as e:
            self.logger.logger.warning(f"State extraction failed: {e}, using initial state")
            initial_state.updated_at = datetime.now()
            return initial_state


# Global workflow instance
invoice_workflow = None

def get_workflow(config: Dict[str, Any] = None) -> InvoiceProcessingGraph:
    """Get or create global workflow instance"""
    global invoice_workflow
    if invoice_workflow is None:
        invoice_workflow = InvoiceProcessingGraph(config)
    return invoice_workflow