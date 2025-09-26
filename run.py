#!/usr/bin/env python3
"""
Simple runner script for Invoice AgenticAI LangGraph system
Starts both the payment API and Streamlit app
"""

import subprocess
import sys
import time
import os
import signal
from threading import Thread

def run_payment_api():
    """Run the payment API server"""
    print("🚀 Starting Payment API server...")
    try:
        subprocess.run([
            sys.executable, "payment_api.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n💳 Payment API server stopped")
    except Exception as e:
        print(f"❌ Payment API failed: {e}")

def run_streamlit_app():
    """Run the Streamlit application"""
    print("🎨 Starting Streamlit application...")
    try:
        subprocess.run([
            "streamlit", "run", "main.py", 
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🎨 Streamlit application stopped")
    except Exception as e:
        print(f"❌ Streamlit failed: {e}")

def check_requirements():
    """Check if required directories and files exist"""
    required_dirs = [
        "data/invoices",
        "logs",
        "output/audit",
        "output/escalations"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"📁 Creating directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    # Check for .env file
    if not os.path.exists(".env"):
        print("⚠️  Warning: .env file not found!")
        print("   Please copy .env.example to .env and add your API keys")
        print("   cp .env.example .env")
        return False
    
    return True

def main():
    """Main runner function"""
    print("🤖 Invoice AgenticAI - LangGraph System")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Setup incomplete. Please fix the issues above and try again.")
        return
    
    print("✅ Setup check passed!")
    print("\nStarting services...")
    print("- Payment API will run on: http://localhost:8000")
    print("- Streamlit app will run on: http://localhost:8501")
    print("\nPress Ctrl+C to stop all services\n")
    
    # Start payment API in background thread
    api_thread = Thread(target=run_payment_api, daemon=True)
    api_thread.start()
    
    # Wait a moment for API to start
    time.sleep(2)
    
    try:
        # Start Streamlit app (blocking)
        run_streamlit_app()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all services...")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("👋 Goodbye!")

if __name__ == "__main__":
    main()