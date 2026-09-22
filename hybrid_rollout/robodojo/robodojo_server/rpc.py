"""Single-owner blocking RPC; disconnects finalize evidence, never retry actions."""
import json
import os
import socket
import traceback
from .protocol import VERSION, receive_packet, send_packet

# How long the simulator waits between a controller's requests. This bounds an
# abandoned controller, not the model: the gap covers the controller's whole
# think time, so a slow local model can exceed the historical 900 s default and
# have its episode closed as `server_close` with zero control steps. Raise it
# with ROBODOJO_RPC_TIMEOUT_SECONDS for such a deployment; it is a transport
# keep-alive only and changes no validation, action or scoring behaviour.
IDLE_TIMEOUT_SECONDS = float(os.environ.get('ROBODOJO_RPC_TIMEOUT_SECONDS', '900'))


def serve(session, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', port))
        listener.listen(1)
        print(json.dumps(dict(event='ready', port=port, metadata=session.metadata)), flush=True)
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(IDLE_TIMEOUT_SECONDS)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                while True:
                    request = receive_packet(connection)
                    response = dict(version=VERSION, request_id=request.get('request_id'))
                    try:
                        if request.get('version') != VERSION:
                            raise ValueError('Unsupported protocol version')
                        result = session.dispatch(request['op'], request.get('args', {}))
                        response.update(ok=True, result=result)
                    except Exception as exc:
                        session.poisoned = True
                        traceback.print_exc()
                        response.update(ok=False, error=f'{type(exc).__name__}: {exc}')
                    send_packet(connection, response)
            except (EOFError, ConnectionError, TimeoutError):
                session._write_summary('terminal' if session.terminated or session.truncated else 'controller_disconnect')
