import sys
import os

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
app_path = os.path.join(src_dir, "gui_app.py")

# Ensure src is in python path for imports
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Run the actual application code
# We use exec() so that Streamlit re-runs the code on interactions
# triggering the app to refresh correctly.
if os.path.exists(app_path):
    with open(app_path, "r", encoding="utf-8") as f:
        code = f.read()
        exec(code, {"__file__": app_path, "__name__": "__main__"})
else:
    print(f"Error: Could not find application at {app_path}")
