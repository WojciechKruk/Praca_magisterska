import paramiko
from scp import SCPClient
import os
import time

# Dane dostępowe
REMOTE_HOST = "192.168.88.88"
REMOTE_PORT = 22
USERNAME = "baumer"
PASSWORD = "amalus0102!"

local_file_path = r"/src/camera_test.py"
remote_file_path = "/home/baumer/camera_test.py"
remote_output_path = "/home/baumer/nagranie_testowe.mp4"
local_output_path = r"C:\Users\krukw\Pobrane\nagranie_testowe.mp4"
remote_log_path = "/home/baumer/exec_log.txt"
local_log_path = r"C:\Users\krukw\Pobrane\exec_log.txt"

def create_ssh_client():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_HOST, port=REMOTE_PORT, username=USERNAME, password=PASSWORD)
    return ssh

try:
    print("Łączenie przez SSH...")
    ssh = create_ssh_client()

    with SCPClient(ssh.get_transport()) as scp:
        print("Wysyłanie pliku na kamerę...")
        scp.put(local_file_path, remote_file_path)

    print("Uruchamianie zdalnego skryptu...")
    command = f"python3.6 {remote_file_path} > {remote_log_path} 2>&1"
    print("Wysyłam polecenie:", command)
    stdin, stdout, stderr = ssh.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()  # czekaj na zakończenie
    print(f"Zdalny skrypt zakończony z kodem: {exit_status}")

    print("Pobieranie logu z kamery...")
    with SCPClient(ssh.get_transport()) as scp:
        scp.get(remote_log_path, local_log_path)

    # Wyświetl log
    with open(local_log_path, "r", encoding="utf-8") as f:
        print("=== ZAWARTOŚĆ LOGU ===")
        print(f.read())

    print("Sprawdzanie zawartości katalogu /home/baumer/")
    stdin, stdout, stderr = ssh.exec_command("ls -l /home/baumer/")
    for line in stdout:
        print(line.strip())

    print(f"Pobieranie pliku wynikowego z kamery: {remote_output_path}...")
    with SCPClient(ssh.get_transport()) as scp:
        scp.get(remote_output_path, local_output_path)

    print("Gotowe!")

except Exception as e:
    print("Wystąpił błąd:", str(e))

finally:
    ssh.close()
