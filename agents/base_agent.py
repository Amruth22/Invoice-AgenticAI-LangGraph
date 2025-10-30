"""
Base Agent Class for Invoice Processing System
Provides common functionality and interface for all specialized agents
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

from state import InvoiceProcessingState, ProcessingStatus, AuditTrail
from utils.logger import get_logger


class BaseAgent(ABC):
    """
    Abstract base class for all invoice processing agents
    Provides common functionality like logging, error handling, and state management
    """
    
    def __init__(self, agent_name: str, config: Dict[str, Any] = None):
        self.agent_name = agent_name
        self.config = config or {}
        self.logger = get_logger(f"agent.{agent_name}")
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        
    @abstractmethod
    async def execute(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """
        Execute the agent's main functionality
        Must be implemented by all concrete agents
        """
        pass
    
    async def run(self, state: InvoiceProcessingState) -> InvoiceProcessingState:
        """
        Main execution wrapper with error handling, logging, and metrics
        """
        start_time = time.time()
        self.execution_count += 1
        
        try:
            self.logger.info(f"Starting {self.agent_name} agent for process {state.process_id}")
            
            # Update state to indicate current agent
            state.current_agent = self.agent_name
            state.add_audit_entry(
                agent_name=self.agent_name,
                action="started",
                status=ProcessingStatus.IN_PROGRESS,
                details={"execution_count": self.execution_count}
            )
            
            # Pre-execution validation
            if not self._validate_preconditions(state):
                raise ValueError(f"Preconditions not met for {self.agent_name}")
            
            # Execute main agent logic
            updated_state = await self.execute(state)
            
            # Post-execution validation
            if not self._validate_postconditions(updated_state):
                raise ValueError(f"Postconditions not met for {self.agent_name}")
            
            # Calculate execution time
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Update success metrics
            self.success_count += 1
            updated_state.update_agent_metrics(self.agent_name, True, duration_ms)
            updated_state.completed_agents.append(self.agent_name)
            
            # Add success audit entry
            updated_state.add_audit_entry(
                agent_name=self.agent_name,
                action="completed",
                status=ProcessingStatus.COMPLETED,
                details={
                    "duration_ms": duration_ms,
                    "success_rate": self.success_count / self.execution_count
                }
            )
            
            self.logger.info(
                f"Completed {self.agent_name} agent for process {state.process_id} "
                f"in {duration_ms}ms"
            )
            
            return updated_state
            
        except Exception as e:
            # Calculate execution time for failed attempt
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Update failure metrics
            self.failure_count += 1
            state.update_agent_metrics(self.agent_name, False, duration_ms)
            
            # Add failure audit entry
            error_message = str(e)
            state.add_audit_entry(
                agent_name=self.agent_name,
                action="failed",
                status=ProcessingStatus.FAILED,
                details={
                    "duration_ms": duration_ms,
                    "error_type": type(e).__name__,
                    "retry_count": state.retry_count
                },
                error_message=error_message
            )
            
            self.logger.error(
                f"Failed {self.agent_name} agent for process {state.process_id}: {error_message}"
            )
            
            # Handle retry logic
            if state.retry_count < state.max_retries:
                state.retry_count += 1
                self.logger.info(f"Scheduling retry {state.retry_count} for {self.agent_name}")
                return state
            else:
                # Max retries reached, escalate
                state.escalation_required = True
                state.escalation_reason = f"Max retries reached for {self.agent_name}: {error_message}"
                state.overall_status = ProcessingStatus.ESCALATED
                return state
    
    def _validate_preconditions(self, state: InvoiceProcessingState) -> bool:
        """
        Validate that the state meets the preconditions for this agent
        Override in concrete agents for specific validation
        """
        return True
    
    def _validate_postconditions(self, state: InvoiceProcessingState) -> bool:
        """
        Validate that the state meets the postconditions after agent execution
        Override in concrete agents for specific validation
        """
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        success_rate = self.success_count / self.execution_count if self.execution_count > 0 else 0
        return {
            "agent_name": self.agent_name,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": success_rate
        }
    
    def reset_metrics(self):
        """Reset agent metrics"""
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for the agent
        Override in concrete agents for specific health checks
        """
        return {
            "agent_name": self.agent_name,
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "metrics": self.get_metrics()
        }
    
    def _extract_business_context(self, state: InvoiceProcessingState) -> Dict[str, Any]:
        """Extract relevant business context for decision making"""
        context = {
            "customer_name": None,
            "invoice_amount": 0.0,
            "due_date": None,
            "priority": state.priority_level,
            "retry_count": state.retry_count
        }
        
        if state.invoice_data:
            context.update({
                "customer_name": state.invoice_data.customer_name,
                "invoice_amount": state.invoice_data.total,
                "due_date": state.invoice_data.due_date,
                "order_id": state.invoice_data.order_id
            })
        
        return context
    
    def _should_escalate(self, state: InvoiceProcessingState, reason: str = None) -> bool:
        """Determine if the current situation requires escalation"""
        escalation_conditions = [
            state.retry_count >= state.max_retries,
            state.escalation_required,
            state.human_review_required
        ]
        
        # Add agent-specific escalation logic
        if hasattr(self, '_agent_specific_escalation'):
            escalation_conditions.append(self._agent_specific_escalation(state))
        
        if any(escalation_conditions) and reason:
            state.escalation_reason = reason
            
        return any(escalation_conditions)
    
    def _log_decision(self, state: InvoiceProcessingState, decision: str, 
                     reasoning: str, confidence: float = None):
        """Log agent decisions for audit trail"""
        details = {
            "decision": decision,
            "reasoning": reasoning,
            "agent_version": getattr(self, 'version', '1.0')
        }
        
        if confidence is not None:
            details["confidence"] = confidence
        
        state.add_audit_entry(
            agent_name=self.agent_name,
            action="decision",
            status=ProcessingStatus.IN_PROGRESS,
            details=details
        )
        
        self.logger.info(f"Decision made by {self.agent_name}: {decision} - {reasoning}")


class AgentRegistry:
    """Registry for managing agent instances"""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
    
    def register(self, agent: BaseAgent):
        """Register an agent instance"""
        self._agents[agent.agent_name] = agent
    
    def get(self, agent_name: str) -> Optional[BaseAgent]:
        """Get an agent by name"""
        return self._agents.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """List all registered agent names"""
        return list(self._agents.keys())
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all registered agents"""
        return {name: agent.get_metrics() for name, agent in self._agents.items()}
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all agents"""
        results = {}
        for name, agent in self._agents.items():
            try:
                results[name] = await agent.health_check()
            except Exception as e:
                results[name] = {
                    "agent_name": name,
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        return results


# Global agent registry instance
agent_registry = AgentRegistry()