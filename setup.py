#!/usr/bin/env python3
"""
Setup script for Invoice AgenticAI LangGraph system
Handles installation, configuration, and initial setup
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("🤖 Invoice AgenticAI - LangGraph Setup")
    print("=" * 60)
    print("Setting up your AI-powered invoice processing system...")
    print()

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible!")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ])
        
        # Install requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        
        print("✅ Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create required directories"""
    print("\n📁 Creating required directories...")
    
    directories = [
        "data/invoices",
        "logs",
        "output/audit",
        "output/escalations",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    print("✅ Directories created successfully!")

def setup_environment():
    """Setup environment configuration"""
    print("\n🔧 Setting up environment configuration...")
    
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            shutil.copy(".env.example", ".env")
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file and add your API keys!")
            print("   Required: GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3, GEMINI_API_KEY_4")
        else:
            print("❌ .env.example not found!")
            return False
    else:
        print("✅ .env file already exists")
    
    return True

def check_api_keys():
    """Check if API keys are configured"""
    print("\n🔑 Checking API key configuration...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        required_keys = [
            "GEMINI_API_KEY_1",
            "GEMINI_API_KEY_2", 
            "GEMINI_API_KEY_3",
            "GEMINI_API_KEY_4"
        ]
        
        missing_keys = []
        for key in required_keys:
            if not os.getenv(key) or os.getenv(key) == f"your_gemini_api_key_{key[-1]}_here":
                missing_keys.append(key)
        
        if missing_keys:
            print("⚠️  Missing or placeholder API keys:")
            for key in missing_keys:
                print(f"   - {key}")
            print("\n   Please get your API keys from Google AI Studio:")
            print("   https://makersuite.google.com/app/apikey")
            return False
        
        print("✅ All API keys configured!")
        return True
        
    except ImportError:
        print("⚠️  python-dotenv not installed, skipping API key check")
        return True

def create_sample_data():
    """Create sample data files if they don't exist"""
    print("\n📄 Setting up sample data...")
    
    # Purchase orders already created
    if os.path.exists("data/purchase_orders.csv"):
        print("✅ Purchase orders data exists")
    else:
        print("❌ Purchase orders data missing!")
        return False
    
    # Check for invoice files
    invoice_dir = Path("data/invoices")
    pdf_files = list(invoice_dir.glob("*.pdf"))
    
    if pdf_files:
        print(f"✅ Found {len(pdf_files)} invoice files")
    else:
        print("⚠️  No invoice PDF files found in data/invoices/")
        print("   Please add some PDF invoice files to test the system")
    
    return True

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        print("   Testing imports...")
        import streamlit
        import pandas
        import plotly
        from langgraph.graph import StateGraph
        import google.generativeai as genai
        print("   ✅ Core imports successful")
        
        # Test file structure
        print("   Testing file structure...")
        required_files = [
            "main.py",
            "payment_api.py",
            "graph/invoice_graph.py",
            "agents/document_agent.py"
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"   ❌ Missing file: {file_path}")
                return False
        
        print("   ✅ File structure valid")
        print("✅ Installation test passed!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. 🔑 Configure your API keys in the .env file")
    print("2. 📄 Add PDF invoice files to data/invoices/ directory")
    print("3. 🚀 Start the system:")
    print("   python run.py")
    print()
    print("Or start services manually:")
    print("   Terminal 1: python payment_api.py")
    print("   Terminal 2: streamlit run main.py")
    print()
    print("🌐 Access the application at: http://localhost:8501")
    print("💳 Payment API will be at: http://localhost:8000")
    print()
    print("📚 For more information, see README.md")
    print("=" * 60)

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Setup failed during environment configuration")
        sys.exit(1)
    
    # Check API keys
    api_keys_ok = check_api_keys()
    
    # Create sample data
    if not create_sample_data():
        print("\n❌ Setup failed during sample data creation")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("\n❌ Setup failed during installation test")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()
    
    if not api_keys_ok:
        print("\n⚠️  Remember to configure your API keys before running the system!")

if __name__ == "__main__":
    main()