"""Small runtime helpers shared by the distributed processes."""
import asyncio
import signal
import sys
import time

def log(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

def install_stop_handlers(stop: asyncio.Event):
    """Set `stop` on SIGINT/SIGTERM (Ctrl+C, docker stop). Works on Windows too."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError, ValueError):
            # Windows
            try:
                signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))
            except (ValueError, OSError):
                pass
