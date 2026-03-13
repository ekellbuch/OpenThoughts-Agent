#!/usr/bin/env python3
"""Minimal SOCKS5 proxy server for Leonardo login nodes.

Leonardo compute nodes cannot SSH back to login nodes (pubkey auth disabled),
so we run a standalone SOCKS5 proxy on the login node instead. Compute nodes
connect via proxychains to login_node:PORT directly over TCP.

Usage (on login node, in a tmux session):
    python3 eval/leonardo/socks5_proxy.py --port 17003

Then in sbatch, proxychains.conf points to login_node:17003.
"""

import argparse
import logging
import os
import select
import signal
import socket
import struct
import sys
import threading

logging.basicConfig(
    level=logging.INFO,
    format="[socks5-proxy] %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BUFFER_SIZE = 65536
CONNECT_TIMEOUT = 15
IDLE_TIMEOUT = 300


def relay(client: socket.socket, remote: socket.socket):
    """Bidirectional relay between client and remote."""
    try:
        while True:
            r, _, _ = select.select([client, remote], [], [], IDLE_TIMEOUT)
            if not r:
                break
            for s in r:
                data = s.recv(BUFFER_SIZE)
                if not data:
                    return
                (remote if s is client else client).sendall(data)
    except (OSError, BrokenPipeError):
        pass
    finally:
        client.close()
        remote.close()


def handle_client(client: socket.socket, addr):
    """Handle one SOCKS5 client connection."""
    try:
        # SOCKS5 greeting
        greeting = client.recv(256)
        if not greeting or greeting[0] != 0x05:
            client.close()
            return
        client.send(b"\x05\x00")  # no auth required

        # SOCKS5 request
        data = client.recv(256)
        if len(data) < 7 or data[1] != 0x01:  # only CONNECT
            client.close()
            return

        atyp = data[3]
        if atyp == 0x01:  # IPv4
            dest_addr = socket.inet_ntoa(data[4:8])
            dest_port = struct.unpack("!H", data[8:10])[0]
        elif atyp == 0x03:  # Domain name
            dl = data[4]
            dest_addr = data[5 : 5 + dl].decode()
            dest_port = struct.unpack("!H", data[5 + dl : 7 + dl])[0]
        elif atyp == 0x04:  # IPv6
            dest_addr = socket.inet_ntop(socket.AF_INET6, data[4:20])
            dest_port = struct.unpack("!H", data[20:22])[0]
        else:
            client.close()
            return

        # Connect to destination
        remote = socket.create_connection((dest_addr, dest_port), timeout=CONNECT_TIMEOUT)

        # Success response
        bind_addr = remote.getsockname()
        reply = b"\x05\x00\x00\x01"
        reply += socket.inet_aton(bind_addr[0])
        reply += struct.pack("!H", bind_addr[1])
        client.send(reply)

        # Relay data
        relay(client, remote)

    except Exception:
        try:
            # Connection refused / failure response
            client.send(b"\x05\x05\x00\x01" + b"\x00" * 6)
            client.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="SOCKS5 proxy for Leonardo")
    parser.add_argument("--port", type=int, default=17003)
    parser.add_argument("--bind", default="0.0.0.0")
    args = parser.parse_args()

    # Write PID file for cleanup
    pid_file = os.path.expanduser("~/.socks5_proxy.pid")
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(256)

    log.info(f"SOCKS5 proxy listening on {args.bind}:{args.port} (PID {os.getpid()})")

    try:
        while True:
            client, addr = srv.accept()
            t = threading.Thread(target=handle_client, args=(client, addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        srv.close()
        os.unlink(pid_file)


if __name__ == "__main__":
    main()
