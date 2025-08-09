#!/usr/bin/env python3
"""
Configuration Profile Manager
Command-line utility to manage LAEF trading configuration profiles
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_manager import config_manager
from datetime import datetime

def list_all_profiles():
    """List all available configuration profiles"""
    print("\n" + "="*60)
    print("LAEF CONFIGURATION PROFILES")
    print("="*60)
    
    if not config_manager.profiles:
        print("No profiles found.")
        return
    
    for profile_key, profile_data in config_manager.profiles.items():
        if not profile_data:
            continue
            
        print(f"\n📁 {profile_key.upper()}")
        print(f"   Name: {profile_data.get('name', 'Unknown')}")
        print(f"   Description: {profile_data.get('description', 'No description')}")
        
        if "created" in profile_data:
            try:
                created = datetime.fromisoformat(profile_data["created"]).strftime("%Y-%m-%d %H:%M")
                print(f"   Created: {created}")
            except:
                print(f"   Created: {profile_data['created']}")
        
        # Show engine information
        if "engine_info" in profile_data:
            engine = profile_data["engine_info"]
            print(f"   Engine: {engine.get('engine_name', 'Unknown')}")
            print(f"   Type: {engine.get('engine_type', 'Unknown')}")
        
        # Show key settings
        if "trading_thresholds" in profile_data:
            thresholds = profile_data["trading_thresholds"]
            print(f"   Settings:")
            print(f"     • Q-Buy: {thresholds.get('q_buy', 'N/A')}")
            print(f"     • Q-Sell: {thresholds.get('q_sell', 'N/A')}")
            print(f"     • Profit Target: {thresholds.get('sell_profit_pct', 0)*100:.1f}%")
            print(f"     • Stop Loss: {(1-thresholds.get('stop_loss_pct', 0.97))*100:.1f}%")
        
        # Show performance data if available
        if "performance_data" in profile_data and profile_data["performance_data"]:
            perf = profile_data["performance_data"]
            if "expected_improvement" in perf:
                print(f"   💰 Expected Improvement: {perf['expected_improvement']}")

def show_profile_details(profile_name):
    """Show detailed information about a specific profile"""
    if profile_name not in config_manager.profiles:
        print(f"Profile '{profile_name}' not found.")
        return
    
    profile_data = config_manager.profiles[profile_name]
    config_manager._show_configuration_preview(profile_data, profile_name.upper())

def delete_profile(profile_name):
    """Delete a configuration profile"""
    if profile_name in ["original", "optimized"]:
        print(f"Cannot delete system profile '{profile_name}'")
        return
    
    if profile_name not in config_manager.profiles:
        print(f"Profile '{profile_name}' not found.")
        return
    
    confirm = input(f"Are you sure you want to delete profile '{profile_name}'? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        # Remove from profiles dict
        del config_manager.profiles[profile_name]
        
        # Remove file if exists
        profile_file = os.path.join(config_manager.config_profiles_dir, f"{profile_name}.json")
        if os.path.exists(profile_file):
            os.remove(profile_file)
            print(f"Deleted profile file: {profile_file}")
        
        # Save updated profiles
        config_manager.save_profiles()
        print(f"Profile '{profile_name}' deleted successfully.")
    else:
        print("Deletion cancelled.")

def create_quick_profile():
    """Create a new profile with minimal input"""
    print("\nQUICK PROFILE CREATION")
    print("-" * 30)
    
    name = input("Profile name: ").strip()
    if not name:
        print("Profile name required.")
        return
    
    description = input("Description (optional): ").strip()
    if not description:
        description = f"Quick profile created on {datetime.now().strftime('%Y-%m-%d')}"
    
    # Get basic settings
    print("\nBasic Settings (press Enter to use defaults):")
    
    q_buy = input("Q-Buy threshold [0.55]: ").strip()
    q_buy = float(q_buy) if q_buy else 0.55
    
    q_sell = input("Q-Sell threshold [0.35]: ").strip()
    q_sell = float(q_sell) if q_sell else 0.35
    
    profit_target = input("Profit target % [4.0]: ").strip()
    profit_target = float(profit_target) / 100 if profit_target else 0.04
    
    stop_loss = input("Stop loss % [3.0]: ").strip()
    stop_loss = 1 - (float(stop_loss) / 100) if stop_loss else 0.97
    
    # Create profile
    profile_data = {
        "name": name,
        "description": description,
        "created": datetime.now().isoformat(),
        "engine_info": {
            "engine_type": "dual_model_engine",
            "engine_name": "LAEF Dual-Model Trading Engine",
            "engine_description": "Q-Learning + ML confidence dual model system",
            "engine_version": "2.0"
        },
        "trading_thresholds": {
            "q_buy": q_buy,
            "q_sell": q_sell,
            "ml_profit_peak": q_buy,  # Same as q_buy
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "sell_profit_pct": profit_target,
            "stop_loss_pct": stop_loss
        },
        "risk_management": {
            "initial_cash": 100000,
            "max_risk_per_trade": 0.025,
            "max_position_size": 0.15,
            "cooldown_minutes": 2
        },
        "backtest_settings": {
            "start_date": "2023-01-01",
            "end_date": "2024-12-31"
        }
    }
    
    # Show preview
    config_manager._show_configuration_preview(profile_data, name.upper())
    
    confirm = input(f"\nSave profile '{name}'? (Y/n): ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        saved_name = config_manager.save_named_profile(name, profile_data)
        if saved_name:
            print(f"✅ Profile '{saved_name}' created successfully!")
        else:
            print("❌ Failed to save profile.")
    else:
        print("Profile creation cancelled.")

def main():
    """Main command-line interface"""
    if len(sys.argv) < 2:
        print("LAEF Configuration Profile Manager")
        print("\nUsage:")
        print("  python profile_manager.py list                    - List all profiles")
        print("  python profile_manager.py show <profile_name>     - Show profile details")
        print("  python profile_manager.py delete <profile_name>   - Delete a profile")
        print("  python profile_manager.py create                  - Create new profile")
        print("  python profile_manager.py interactive             - Interactive mode")
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        list_all_profiles()
    
    elif command == "show":
        if len(sys.argv) < 3:
            print("Usage: python profile_manager.py show <profile_name>")
            return
        profile_name = sys.argv[2]
        show_profile_details(profile_name)
    
    elif command == "delete":
        if len(sys.argv) < 3:
            print("Usage: python profile_manager.py delete <profile_name>")
            return
        profile_name = sys.argv[2]
        delete_profile(profile_name)
    
    elif command == "create":
        create_quick_profile()
    
    elif command == "interactive":
        interactive_mode()
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'python profile_manager.py' for help")

def interactive_mode():
    """Interactive profile management"""
    while True:
        print("\n" + "="*50)
        print("PROFILE MANAGER - INTERACTIVE MODE")
        print("="*50)
        print("1. List all profiles")
        print("2. Show profile details")
        print("3. Create new profile")
        print("4. Delete profile")
        print("5. Test configuration")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            list_all_profiles()
        elif choice == "2":
            profile_name = input("Profile name: ").strip()
            if profile_name:
                show_profile_details(profile_name)
        elif choice == "3":
            create_quick_profile()
        elif choice == "4":
            profile_name = input("Profile name to delete: ").strip()
            if profile_name:
                delete_profile(profile_name)
        elif choice == "5":
            print("Starting configuration selection...")
            try:
                config = config_manager.select_configuration_interactive()
                if config:
                    print("✅ Configuration selected successfully!")
                    print(f"Selected profile: {config.get('_profile_name', 'Unknown')}")
                else:
                    print("❌ No configuration selected.")
            except Exception as e:
                print(f"❌ Error: {e}")
        elif choice == "6":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()