import argparse
import os
import subprocess
import sys
import time

def get_python_exe() -> str:
    """Returns path to the current Python interpreter."""
    return sys.executable


def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """Launches the FastAPI backend server."""
    print(f"🚀 Starting FastAPI backend on http://{host}:{port}...")
    cmd = [
        get_python_exe(),
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        cmd.append("--reload")
    subprocess.run(cmd)


def run_ui(port: int = 8501):
    """Launches the Streamlit chat frontend."""
    print(f"🌐 Starting Streamlit Chat UI on http://localhost:{port}...")
    cmd = [
        get_python_exe(),
        "-m",
        "streamlit",
        "run",
        "frontend/streamlit_app.py",
        "--server.port",
        str(port),
    ]
    subprocess.run(cmd)


def run_server():
    """Runs the MCP server over stdio transport."""
    print("🛰️ Starting Wikipedia MCP server in stdio mode...")
    cmd = [get_python_exe(), "mcp_server/server.py"]
    subprocess.run(cmd)


def run_tests():
    """Runs the pytest test suite."""
    print("🧪 Running test suite...")
    cmd = [get_python_exe(), "-m", "pytest", "tests/", "-v"]
    subprocess.run(cmd)


def run_all(api_port: int = 8000, ui_port: int = 8501):
    """Runs FastAPI backend and Streamlit UI concurrently."""
    print("================================================================")
    print("  Starting Wikipedia MCP Agent (Backend + Frontend)")
    print(f"  FastAPI Docs:   http://localhost:{api_port}/docs")
    print(f"  Streamlit Chat: http://localhost:{ui_port}")
    print("================================================================")

    api_cmd = [
        get_python_exe(),
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(api_port),
    ]
    api_process = subprocess.Popen(api_cmd)

    time.sleep(2)

    env = dict(os.environ)
    env["BACKEND_URL"] = f"http://localhost:{api_port}"
    ui_cmd = [
        get_python_exe(),
        "-m",
        "streamlit",
        "run",
        "frontend/streamlit_app.py",
        "--server.port",
        str(ui_port),
    ]
    ui_process = subprocess.Popen(ui_cmd, env=env)

    try:
        api_process.wait()
        ui_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down backend and frontend...")
        api_process.terminate()
        ui_process.terminate()
        api_process.wait()
        ui_process.wait()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Wikipedia MCP Agent Application Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser("api", help="Run FastAPI backend")
    subparsers.add_parser("ui", help="Run Streamlit Chat UI")
    subparsers.add_parser("server", help="Run MCP Server directly")
    subparsers.add_parser("test", help="Run unit & integration tests")
    subparsers.add_parser("all", help="Run both FastAPI backend and Streamlit UI")

    args = parser.parse_args()

    if args.command == "api":
        run_api()
    elif args.command == "ui":
        run_ui()
    elif args.command == "server":
        run_server()
    elif args.command == "test":
        run_tests()
    elif args.command == "all":
        run_all()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
