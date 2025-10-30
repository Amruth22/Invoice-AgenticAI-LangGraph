"""
Document Processing Node
Extracts and processes invoice documents through the DocumentAgent.
"""

from state import InvoiceProcessingState
from agents.document_agent import DocumentAgent

# Initialize agent at module level with default config
document_agent = DocumentAgent({})


async def document_processing_node(state: InvoiceProcessingState) -> InvoiceProcessingState:
    """
    Document Processing Node - extracts and processes invoice documents.

    This node is executed by the DocumentAgent through the workflow.
    It handles PDF extraction, text parsing, and initial data collection.
    """
    return await document_agent.run(state)
