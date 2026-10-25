"""CONNECT-прокси стенда: единственный разрешённый узел — шлюз организаторов.

TLS не расшифровывается; ключи и содержимое запросов не журналируются.
"""
import select
import socket
import socketserver
import time

ALLOWED_HOST = "api.alfahack-gateway26.ru"


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(10)
        first = self.rfile.readline(4097)
        try:
            method, destination, protocol = first.decode("ascii").strip().split()
        except (ValueError, UnicodeDecodeError):
            return
        total = len(first)
        while True:
            line = self.rfile.readline(4097)
            total += len(line)
            if total > 16000:
                return
            if line in (b"\r\n", b"\n", b""):
                break
        if method != "CONNECT" or destination.lower() != ALLOWED_HOST+":443":
            self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            self.wfile.flush()
            print("DENIED", destination[:200], flush=True)
            return
        remote = None
        try:
            remote = socket.create_connection((ALLOWED_HOST,443),timeout=15)
            self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            self.wfile.flush()
            print("ALLOWED",ALLOWED_HOST,flush=True)
            sockets = [self.connection,remote]
            end = time.monotonic()+120
            while time.monotonic()<end:
                readable,_,_ = select.select(sockets,[],[],2)
                for source in readable:
                    block = source.recv(65536)
                    if not block:
                        return
                    target = remote if source is self.connection else self.connection
                    target.sendall(block)
        except (OSError,TimeoutError):
            return
        finally:
            if remote is not None:
                remote.close()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    with Server(("0.0.0.0",8080),Handler) as server:
        print("Шлюз стенда: разрешён только "+ALLOWED_HOST,flush=True)
        server.serve_forever()
