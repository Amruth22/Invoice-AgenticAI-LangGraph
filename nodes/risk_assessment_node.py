"""
Risk Assessment Node
Performs risk assessment on invoices through the RiskAgent.
"""

from state import InvoiceProcessingState
from agents.risk_agent import RiskAgent

# Initialize agent at module level
risk_agent = RiskAgent({})


async def risk_assessment_node(state: InvoiceProcessingState) -> InvoiceProcessingState:
    """
    Risk Assessment Node - performs risk assessment on invoices.

    This node is executed by the RiskAgent through the workflow.
    It analyzes payment risk, fraud detection, and flags high-risk invoices.
    """
    return await risk_agent.run(state)
