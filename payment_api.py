"""
FastAPI Payment Simulation Service
Simulates payment processing for the invoice automation system
"""

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from datetime import datetime
import random
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Invoice Payment API",
    description="Payment processing simulation for invoice automation",
    version="1.0.0"
)

class PaymentRequest(BaseModel):
    order_id: str
    customer_name: str
    amount: float
    due_date: str
    payment_method: str = "ach"
    invoice_number: str = None

class PaymentResponse(BaseModel):
    transaction_id: str
    status: str
    timestamp: str
    message: str
    payment_method: str
    processing_fee: float = 0.0

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Invoice Payment API",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": "running",
        "database": "simulated",
        "payment_gateway": "simulated"
    }

@app.post("/initiate_payment", response_model=PaymentResponse)
async def initiate_payment(payment_request: PaymentRequest):
    """
    Initiate payment processing
    Simulates real payment gateway behavior with various scenarios
    """
    logger.info(f"Processing payment for {payment_request.customer_name}: ${payment_request.amount}")
    
    # Simulate processing delay
    import time
    time.sleep(0.1)  # Small delay to simulate processing
    
    # Generate transaction ID
    transaction_id = f"TXN-{random.randint(100000, 999999)}"
    
    # Simulate different payment scenarios based on amount and customer
    if payment_request.amount > 100000:
        # Very high amounts might fail
        if random.random() < 0.1:  # 10% failure rate for high amounts
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "AMOUNT_LIMIT_EXCEEDED",
                    "message": f"Amount ${payment_request.amount} exceeds daily limit",
                    "transaction_id": transaction_id
                }
            )
    
    # Simulate customer-specific scenarios
    if "test" in payment_request.customer_name.lower():
        # Test customers always succeed
        status = "SUCCESS"
        message = f"Test payment processed successfully for {payment_request.customer_name}"
    elif payment_request.amount < 1.0:
        # Very small amounts might be rejected
        status = "REJECTED"
        message = "Amount too small for processing"
    else:
        # Normal processing with high success rate
        if random.random() < 0.95:  # 95% success rate
            status = "SUCCESS"
            message = f"Payment processed successfully for {payment_request.customer_name}"
        else:
            status = "PENDING_REVIEW"
            message = "Payment requires additional verification"
    
    # Calculate processing fee based on payment method
    processing_fees = {
        "ach": 0.50,
        "wire": 15.00,
        "card": payment_request.amount * 0.029,  # 2.9%
        "check": 2.00
    }
    
    processing_fee = processing_fees.get(payment_request.payment_method, 1.00)
    
    response = PaymentResponse(
        transaction_id=transaction_id,
        status=status,
        timestamp=datetime.utcnow().isoformat(),
        message=message,
        payment_method=payment_request.payment_method,
        processing_fee=processing_fee
    )
    
    logger.info(f"Payment result: {status} - {transaction_id}")
    
    return response

@app.get("/transaction/{transaction_id}")
async def get_transaction_status(transaction_id: str):
    """Get transaction status by ID"""
    # Simulate transaction lookup
    if not transaction_id.startswith("TXN-"):
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    # Simulate various transaction states
    statuses = ["SUCCESS", "PENDING", "FAILED", "CANCELLED"]
    status = random.choice(statuses)
    
    return {
        "transaction_id": transaction_id,
        "status": status,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "amount": random.uniform(100, 10000),
        "currency": "USD"
    }

@app.post("/cancel_payment/{transaction_id}")
async def cancel_payment(transaction_id: str):
    """Cancel a pending payment"""
    if not transaction_id.startswith("TXN-"):
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return {
        "transaction_id": transaction_id,
        "status": "CANCELLED",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "Payment cancelled successfully"
    }

@app.get("/payment_methods")
async def get_payment_methods():
    """Get available payment methods and their limits"""
    return {
        "payment_methods": [
            {
                "method": "ach",
                "name": "ACH Transfer",
                "limit": 100000,
                "processing_days": 3,
                "fee": 0.50
            },
            {
                "method": "wire",
                "name": "Wire Transfer",
                "limit": 1000000,
                "processing_days": 1,
                "fee": 15.00
            },
            {
                "method": "card",
                "name": "Credit Card",
                "limit": 10000,
                "processing_days": 1,
                "fee_percentage": 2.9
            },
            {
                "method": "check",
                "name": "Check",
                "limit": 50000,
                "processing_days": 5,
                "fee": 2.00
            }
        ]
    }

@app.get("/metrics")
async def get_metrics():
    """Get payment processing metrics"""
    return {
        "total_transactions": random.randint(1000, 5000),
        "successful_payments": random.randint(900, 1000),
        "failed_payments": random.randint(10, 50),
        "pending_payments": random.randint(5, 20),
        "total_volume": random.uniform(100000, 1000000),
        "average_amount": random.uniform(1000, 5000),
        "uptime_percentage": 99.9,
        "last_updated": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "payment_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )