import socket
import struct

def main():
    server_ip = '0.0.0.0'  # 監聽所有網路介面
    server_port = 8080     # 與 client 相同的 port

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((server_ip, server_port))
    print(f"UDP 伺服器正在監聽 {server_ip}:{server_port}")

    try:
        while True:
            # 接收封包資料（假設封包不會超過 65535 bytes）
            data, addr = sock.recvfrom(65535)
            if not data:
                continue
            print(f"從 {addr} 收到封包，大小 {len(data)} bytes")
    
            # 檢查資料至少包含 4 bytes（點數資訊）
            if len(data) < 4:
                print("資料長度不足")
                continue
    
            # 讀取前 4 bytes 作為點數 (uint32 little-endian)
            num_points = struct.unpack('<I', data[:4])[0]
            print(f"封包中包含 {num_points} 個點")
    
            expected_size = 4 + num_points * 3 * 4
            if len(data) != expected_size:
                print(f"資料長度與預期不符，預期 {expected_size} bytes")
                continue
    
            # 從資料中解析出所有 float 值
            floats = struct.unpack('<' + 'f' * (num_points * 3), data[4:])
            points = [(floats[i], floats[i+1], floats[i+2])
                      for i in range(0, len(floats), 3)]
    
            # 示範：僅印出前 3 個點
            print(f"接收到 {len(points)} 個點，範例: {points[:3]}")
    except KeyboardInterrupt:
        print("伺服器中斷")
    finally:
        sock.close()

if __name__ == '__main__':
    main()
