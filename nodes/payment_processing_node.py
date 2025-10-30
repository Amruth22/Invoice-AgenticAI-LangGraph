"""
Payment Processing Node
Processes and manages invoice payments through the PaymentAgent.
"""

from state import InvoiceProcessingState
from agents.payment_agent import PaymentAgent

# Initialize agent at module level
payment_agent = PaymentAgent({})


async def payment_processing_node(state: InvoiceProcessingState) -> InvoiceProcessingState:
    """
    Payment Processing Node - processes and manages invoice payments.

    This node is executed by the PaymentAgent through the workflow.
    It handles payment scheduling, processing, and status tracking.
    """
    return await payment_agent.run(state)
