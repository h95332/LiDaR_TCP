import socket, struct

HOST = '127.0.0.1'   # 本機回環
PORT = 9000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"伺服器啟動，監聽 {HOST}:{PORT}")
    conn, addr = srv.accept()
    with conn:
        print("Client 已連線：", addr)
        while True:
            header = conn.recv(8)
            if not header:
                print("連線關閉"); break
            coord_len, pts_len = struct.unpack('<II', header)
            coord = conn.recv(coord_len)
            pts   = conn.recv(pts_len)
            print(f"收到 → coord {coord_len} bytes, points {pts_len} bytes")