#!/usr/bin/env python3
"""
Simple HTTP server for the results viewer
"""
import http.server
import socketserver
import webbrowser
from pathlib import Path
import socket

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

class ReusableTCPServer(socketserver.TCPServer):
    """TCP Server with SO_REUSEADDR and SO_REUSEPORT to allow port reuse"""
    allow_reuse_address = True
    
    def server_bind(self):
        # Set SO_REUSEPORT in addition to SO_REUSEADDR for better reuse
        if hasattr(socket, 'SO_REUSEPORT'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        super().server_bind()

def main():
    # Change to the project directory
    project_dir = Path(__file__).parent
    import os
    os.chdir(project_dir)
    
    Handler = MyHTTPRequestHandler
    
    with ReusableTCPServer(("", PORT), Handler) as httpd:
        print("=" * 60)
        print(f"🚀 Results Viewer Server Running")
        print("=" * 60)
        print(f"📊 View your results at: http://localhost:{PORT}/results_viewer.html")
        print(f"📁 Serving from: {project_dir}")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 60)
        
        # Try to open browser automatically
        try:
            webbrowser.open(f'http://localhost:{PORT}/results_viewer.html')
        except:
            pass
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped")

if __name__ == "__main__":
    main()

