"""
Validation Node
Validates extracted invoice data through the ValidationAgent.
"""

from state import InvoiceProcessingState
from agents.validation_agent import ValidationAgent

# Initialize agent at module level
validation_agent = ValidationAgent({})


async def validation_node(state: InvoiceProcessingState) -> InvoiceProcessingState:
    """
    Validation Node - validates extracted invoice data.

    This node is executed by the ValidationAgent through the workflow.
    It performs data quality checks and identifies missing/invalid fields.
    """
    return await validation_agent.run(state)
