# === check_dependencies.py - Verify LAEF Dependencies ===

import sys
import importlib
from importlib import metadata

def check_package(package_name, import_name=None, min_version=None):
    """Check if a package is installed and meets version requirements"""
    if import_name is None:
        import_name = package_name
    
    try:
        # Try to import the package
        module = importlib.import_module(import_name)
        
        # Get version
        try:
            version = metadata.version(package_name)
        except:
            version = getattr(module, '__version__', 'Unknown')
        
        status = "✅"
        message = f"{package_name} {version}"
        
        # Check minimum version if specified
        if min_version and version != 'Unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                status = "⚠️"
                message += f" (minimum {min_version} recommended)"
        
        return True, f"{status} {message}"
        
    except ImportError:
        return False, f"❌ {package_name} - NOT INSTALLED"
    except Exception as e:
        return False, f"❌ {package_name} - Error: {str(e)}"

def main():
    print("🔍 LAEF DEPENDENCY CHECK")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print("=" * 60)
    
    # Define packages to check (package_name, import_name, min_version)
    essential_packages = [
        ("pandas", None, "2.0.0"),
        ("numpy", None, "1.24.0"),
        ("tensorflow", None, "2.13.0"),
        ("keras", None, "2.13.0"),
        ("alpaca-trade-api", "alpaca_trade_api", "3.0.0"),
        ("yfinance", None, "0.2.28"),
        ("pytz", None, "2023.3"),
        ("python-dotenv", "dotenv", "1.0.0"),
        ("matplotlib", None, "3.7.0"),
        ("seaborn", None, "0.12.0"),
    ]
    
    optional_packages = [
        ("ta-lib", "talib", None),
        ("plotly", None, "5.14.0"),
        ("jupyter", None, None),
        ("openpyxl", None, None),
        ("tqdm", None, None),
    ]
    
    # Check essential packages
    print("\n📦 ESSENTIAL PACKAGES:")
    print("-" * 40)
    essential_ok = True
    for package_info in essential_packages:
        success, message = check_package(*package_info)
        print(message)
        if not success:
            essential_ok = False
    
    # Check optional packages
    print("\n📦 OPTIONAL PACKAGES:")
    print("-" * 40)
    for package_info in optional_packages:
        success, message = check_package(*package_info)
        print(message)
    
    # Check LAEF specific files
    print("\n📁 LAEF SYSTEM FILES:")
    print("-" * 40)
    
    critical_files = [
        "config.py",
        "agent_unified.py",
        "data_fetcher_unified.py",
        "dual_model_trading_logic.py",
        "fifo_portfolio.py",
        ".env"
    ]
    
    import os
    for file in critical_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - NOT FOUND")
    
    # Summary
    print("\n" + "=" * 60)
    if essential_ok:
        print("✅ All essential packages are installed!")
        print("🚀 Your LAEF system is ready to run!")
    else:
        print("❌ Some essential packages are missing!")
        print("📋 Install missing packages with:")
        print("   pip install -r requirements-minimal.txt")
    
    print("\n💡 For full functionality, install all packages:")
    print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
