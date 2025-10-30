"""
Escalation Node
Handles escalation of problematic invoices through the EscalationAgent.
"""

from state import InvoiceProcessingState
from agents.escalation_agent import EscalationAgent

# Initialize agent at module level
escalation_agent = EscalationAgent({})


async def escalation_node(state: InvoiceProcessingState) -> InvoiceProcessingState:
    """
    Escalation Node - handles escalation of problematic invoices.

    This node is executed by the EscalationAgent through the workflow.
    It identifies issues requiring human review and manages escalations.
    """
    return await escalation_agent.run(state)
