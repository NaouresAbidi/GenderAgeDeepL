#!/usr/bin/env python3
"""Run the API from project root"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the API
from api.api import app

if __name__ == "__main__":
    print("🚀 Starting Age & Gender Prediction API...")
    print("📍 Server will be available at:")
    print("   - Local: http://127.0.0.1:5000")
    print("   - Network: http://0.0.0.0:5000")
    print("📋 Available endpoints:")
    print("   - GET /health - Model status")
    print("   - POST /predict - Age & gender prediction")
    print("   - GET / - API documentation")
    print("\n🛑 Press CTRL+C to stop the server")
    
    app.run(host="0.0.0.0", port=5000, debug=False)