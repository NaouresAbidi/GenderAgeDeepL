#!/usr/bin/env python3
"""Test Flask installation"""

try:
    import flask
    print(f"✅ Flask {flask.__version__} imported successfully")
    
    # Test basic Flask functionality
    from flask import Flask, request, jsonify
    print("✅ Flask components imported successfully")
    
    # Create a test app
    app = Flask(__name__)
    print("✅ Flask app created successfully")
    
    print("\n🎉 Flask is working correctly!")
    
except ImportError as e:
    print(f"❌ Flask import error: {e}")
except Exception as e:
    print(f"❌ Flask error: {e}")