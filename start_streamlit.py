import subprocess
import sys
from src.core.config import settings

if __name__ == "__main__":
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "streamlit_app/app.py",
        "--server.port", str(settings.STREAMLIT_PORT),
        "--server.address", settings.API_HOST
    ]
    subprocess.run(cmd)
