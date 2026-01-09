import os
import re

def clean_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    start_idx = 0
    end_idx = len(lines)
    
    # Common markers
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
        "***START OF THIS PROJECT GUTENBERG EBOOK"
    ]
    
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THIS PROJECT GUTENBERG EBOOK"
    ]
    
    # Find start
    for i, line in enumerate(lines[:500]): # Search first 500 lines
        if any(m in line for m in start_markers):
            start_idx = i + 1
            break
            
    # Find end
    for i, line in enumerate(reversed(lines[-500:])): # Search last 500 lines
        if any(m in line for m in end_markers):
            end_idx = len(lines) - i - 1
            break
            
    # If no markers found, we might want to apply fallback or skip.
    # For now, we only trim if markers are found to be safe.
    
    if start_idx == 0 and end_idx == len(lines):
        # Check for "End of the Project Gutenberg EBook" alternative
        pass

    new_content = lines[start_idx:end_idx]
    
    # Overwrite only if changed
    if len(new_content) < len(lines):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_content)
        return True
    return False

def main():
    root_dir = "data_gutenberg"
    count = 0
    cleaned = 0
    
    print(f"Cleaning files in {root_dir}...")
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                if clean_file(path):
                    cleaned += 1
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} files...")
                    
    print(f"Done. Processed {count} files. Cleaned {cleaned} files.")

if __name__ == "__main__":
    main()
