"""Start SoluAI on this computer and open it in the browser.

    python run.py            # then use http://127.0.0.1:8000
    python run.py --port 9000 --no-browser

Everything the website needs (model, fonts, charts, drawing editor) is in this
folder, so it works without internet. Only looking up *unusual* molecule names
uses the internet (PubChem); the 225 built-in names, SMILES and salts do not.
"""

from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SoluAI website locally.")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to let other devices on your Wi-Fi open it")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    args = parser.parse_args()
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(2.5, webbrowser.open, args=(url,)).start()
    print(f"SoluAI is starting at {url}  (press Ctrl+C to stop)")
    uvicorn.run("web.main:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
