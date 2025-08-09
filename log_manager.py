"""
Log Management Utility for LAEF Trading System
Helps manage large log files by providing analysis, cleanup, and rotation options
"""

import os
import shutil
from datetime import datetime, timedelta
import gzip
import re

class LogManager:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        
    def get_file_size_readable(self, size_bytes):
        """Convert bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    def analyze_log_file(self, filename="main.log"):
        """Analyze log file and show statistics"""
        filepath = os.path.join(self.log_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Log file not found: {filepath}")
            return
            
        file_size = os.path.getsize(filepath)
        print(f"\nLog File Analysis: {filename}")
        print(f"Size: {self.get_file_size_readable(file_size)}")
        
        # Sample the file to get statistics
        print("\nSampling first 1000 lines...")
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                for i, line in enumerate(f):
                    if i < 1000:
                        lines.append(line)
                    else:
                        break
                        
            # Count log levels
            levels = {'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'DEBUG': 0}
            for line in lines:
                for level in levels:
                    if f'[{level}]' in line:
                        levels[level] += 1
                        
            print(f"\nLog levels in sample:")
            for level, count in levels.items():
                print(f"  {level}: {count}")
                
        except Exception as e:
            print(f"Error analyzing file: {e}")
    
    def extract_recent_logs(self, filename="main.log", hours=24, output_file="recent_logs.txt"):
        """Extract only recent log entries"""
        filepath = os.path.join(self.log_dir, filename)
        output_path = os.path.join(self.log_dir, output_file)
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        print(f"\nExtracting logs from last {hours} hours...")
        line_count = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
                with open(output_path, 'w', encoding='utf-8') as outfile:
                    for line in infile:
                        # Try to parse timestamp
                        try:
                            # Look for timestamp pattern: YYYY-MM-DD HH:MM:SS
                            match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                            if match:
                                timestamp_str = match.group(1)
                                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                                
                                if timestamp >= cutoff_time:
                                    outfile.write(line)
                                    line_count += 1
                        except:
                            # If can't parse timestamp, skip line
                            pass
                            
            print(f"Extracted {line_count} lines to {output_file}")
            output_size = os.path.getsize(output_path)
            print(f"Output file size: {self.get_file_size_readable(output_size)}")
            
        except Exception as e:
            print(f"Error extracting logs: {e}")
    
    def extract_errors_only(self, filename="main.log", output_file="errors_only.log"):
        """Extract only ERROR and WARNING messages"""
        filepath = os.path.join(self.log_dir, filename)
        output_path = os.path.join(self.log_dir, output_file)
        
        print(f"\nExtracting errors and warnings...")
        line_count = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
                with open(output_path, 'w', encoding='utf-8') as outfile:
                    for line in infile:
                        if '[ERROR]' in line or '[WARNING]' in line:
                            outfile.write(line)
                            line_count += 1
                            
            print(f"Extracted {line_count} error/warning lines to {output_file}")
            output_size = os.path.getsize(output_path)
            print(f"Output file size: {self.get_file_size_readable(output_size)}")
            
        except Exception as e:
            print(f"Error extracting errors: {e}")
    
    def rotate_log(self, filename="main.log", compress=True):
        """Rotate the log file with timestamp"""
        filepath = os.path.join(self.log_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Log file not found: {filepath}")
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_name = f"{filename}.{timestamp}"
        new_path = os.path.join(self.log_dir, new_name)
        
        try:
            # Move the current log file
            shutil.move(filepath, new_path)
            print(f"Rotated log to: {new_name}")
            
            # Compress if requested
            if compress:
                print("Compressing old log file...")
                with open(new_path, 'rb') as f_in:
                    with gzip.open(f"{new_path}.gz", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove uncompressed file
                os.remove(new_path)
                print(f"Compressed to: {new_name}.gz")
                
                compressed_size = os.path.getsize(f"{new_path}.gz")
                print(f"Compressed size: {self.get_file_size_readable(compressed_size)}")
                
            # Create new empty log file
            open(filepath, 'a').close()
            print(f"Created new empty {filename}")
            
        except Exception as e:
            print(f"Error rotating log: {e}")
    
    def cleanup_old_logs(self, days=7):
        """Remove log files older than specified days"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        print(f"\nCleaning up logs older than {days} days...")
        removed_count = 0
        removed_size = 0
        
        for filename in os.listdir(self.log_dir):
            if filename.endswith('.log') or filename.endswith('.gz'):
                filepath = os.path.join(self.log_dir, filename)
                
                # Check file modification time
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if file_time < cutoff_time and filename != 'main.log':
                    size = os.path.getsize(filepath)
                    os.remove(filepath)
                    removed_count += 1
                    removed_size += size
                    print(f"  Removed: {filename}")
                    
        print(f"Removed {removed_count} files, freed {self.get_file_size_readable(removed_size)}")
    
    def tail_log(self, filename="main.log", lines=100):
        """Show last N lines of log file"""
        filepath = os.path.join(self.log_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Log file not found: {filepath}")
            return
            
        print(f"\nLast {lines} lines of {filename}:")
        print("-" * 80)
        
        try:
            # For very large files, read from end
            with open(filepath, 'rb') as f:
                # Go to end of file
                f.seek(0, 2)
                file_length = f.tell()
                
                # Read backwards to find lines
                block_size = 1024
                blocks = []
                lines_found = 0
                
                while lines_found < lines and file_length > 0:
                    read_size = min(block_size, file_length)
                    f.seek(file_length - read_size)
                    block = f.read(read_size)
                    
                    lines_in_block = block.count(b'\n')
                    lines_found += lines_in_block
                    
                    blocks.insert(0, block)
                    file_length -= read_size
                
                # Join blocks and decode
                all_lines = b''.join(blocks).decode('utf-8', errors='ignore').split('\n')
                
                # Show last N lines
                for line in all_lines[-lines:]:
                    if line.strip():
                        print(line)
                        
        except Exception as e:
            print(f"Error reading log: {e}")

def main():
    """Interactive log management menu"""
    manager = LogManager()
    
    while True:
        print("\n" + "="*60)
        print("LAEF Log Manager")
        print("="*60)
        print("1. Analyze main.log file")
        print("2. Extract recent logs (last 24 hours)")
        print("3. Extract errors and warnings only")
        print("4. Show last 100 lines")
        print("5. Rotate and compress current log")
        print("6. Cleanup old log files")
        print("7. Custom extraction")
        print("8. Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == '1':
            manager.analyze_log_file()
        elif choice == '2':
            manager.extract_recent_logs()
        elif choice == '3':
            manager.extract_errors_only()
        elif choice == '4':
            manager.tail_log()
        elif choice == '5':
            confirm = input("This will rotate main.log. Continue? (y/n): ")
            if confirm.lower() == 'y':
                manager.rotate_log()
        elif choice == '6':
            days = input("Remove logs older than how many days? (default 7): ").strip()
            days = int(days) if days else 7
            manager.cleanup_old_logs(days)
        elif choice == '7':
            hours = input("Extract logs from last how many hours? (default 24): ").strip()
            hours = int(hours) if hours else 24
            output = input("Output filename (default: custom_extract.log): ").strip()
            output = output if output else "custom_extract.log"
            manager.extract_recent_logs(hours=hours, output_file=output)
        elif choice == '8':
            print("Exiting log manager...")
            break
        else:
            print("Invalid option")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()