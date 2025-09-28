"""Output capture utility for real-time console feedback in Streamlit"""
import sys
import io
from typing import Optional, Callable
import threading
import queue
import time


class StreamCapture:
    """Captures stdout and provides real-time updates"""

    def __init__(self, update_callback: Optional[Callable] = None):
        self.update_callback = update_callback
        self.captured_output = []
        self.original_stdout = sys.stdout
        self.capture_buffer = io.StringIO()
        self.queue = queue.Queue()
        self.capturing = False
        self.thread = None

    def start_capture(self):
        """Start capturing stdout"""
        self.capturing = True
        sys.stdout = self

        # Start background thread to process output
        self.thread = threading.Thread(target=self._process_output, daemon=True)
        self.thread.start()

    def stop_capture(self):
        """Stop capturing and restore original stdout"""
        self.capturing = False
        sys.stdout = self.original_stdout
        if self.thread:
            self.thread.join(timeout=1)

    def write(self, text):
        """Intercept stdout writes"""
        # Write to original stdout for debugging
        self.original_stdout.write(text)

        # Add to queue for processing
        if text and text != '\n':
            self.queue.put(text)

    def flush(self):
        """Flush implementation for stdout compatibility"""
        self.original_stdout.flush()

    def _process_output(self):
        """Process captured output in background"""
        while self.capturing:
            try:
                # Get output from queue with timeout
                text = self.queue.get(timeout=0.1)

                # Store in captured output
                self.captured_output.append(text)

                # Call update callback if provided
                if self.update_callback:
                    self.update_callback(text)

            except queue.Empty:
                continue

    def get_output(self):
        """Get all captured output"""
        return ''.join(self.captured_output)

    def get_recent_output(self, lines: int = 5):
        """Get recent output lines"""
        full_output = self.get_output()
        output_lines = full_output.split('\n')
        return '\n'.join(output_lines[-lines:])

    def clear(self):
        """Clear captured output"""
        self.captured_output = []