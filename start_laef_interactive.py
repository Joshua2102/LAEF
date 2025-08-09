#!/usr/bin/env python3
"""
Start LAEF Interactive System
"""

from laef_unified_system import LAEFUnifiedSystem

def main():
    print("Starting LAEF Interactive System...")
    print("Press Ctrl+C to exit at any time")
    print()
    
    try:
        system = LAEFUnifiedSystem()
        system.run()
    except KeyboardInterrupt:
        print("\nExiting LAEF system...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()