#!/bin/bash
# Pre-install numpy first to avoid setup.py build issues
pip install --no-cache-dir numpy==1.26.4

# Now install the rest of your requirements
pip install --no-cache-dir -r requirements.txt
