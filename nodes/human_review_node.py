"""
Human Review Node
Manages human review workflow for escalated invoices.
"""

from state import InvoiceProcessingState


async def human_review_node(state: InvoiceProcessingState) -> InvoiceProcessingState:
    """
    Human Review Node - manages human review workflow for escalated invoices.

    This node handles cases requiring human intervention, decision tracking,
    and integration with manual review processes.
    """
    # Mark that human review was requested
    state.requires_human_review = True

    return state
