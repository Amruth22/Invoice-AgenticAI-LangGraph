"""
Audit Node
Creates audit trail and records all processing activities through the AuditAgent.
"""

from state import InvoiceProcessingState
from agents.audit_agent import AuditAgent

# Initialize agent at module level
audit_agent = AuditAgent({})


async def audit_node(state: InvoiceProcessingState) -> InvoiceProcessingState:
    """
    Audit Node - creates audit trail and records processing activities.

    This node is executed by the AuditAgent through the workflow.
    It logs all decisions, changes, and actions for compliance and tracking.
    """
    return await audit_agent.run(state)
