import os
import time
import json
import requests
import subprocess
from pathlib import Path


class DiskMonitor:
    # 상수 정의
    STRETCH_INTERVAL = 4 * 3600  # 스트레칭 알림 간격: 4시간 (초 단위)
    MONITOR_INTERVAL = 600        # 디스크 모니터링 간격: 10분 (초 단위)
    SSH_TIMEOUT = 30              # SSH 명령어 타임아웃: 30초
    SLACK_TIMEOUT = 10            # Slack 요청 타임아웃: 10초

    def __init__(self):
        # Slack 웹훅 URL
        self.slack_webhook_url = "https://hooks.slack.com/services/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"

        # 서버 정보 딕셔너리
        # ssh -i [키파일 경로] -p [포트넘버] root@[서버주소]
        self.servers = {
            "server1": ["/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/hanseo/diskbot/keykey.pem", "32454", "10.28.224.217"],
            "server2": ["/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/hanseo/diskbot/keykey.pem", "30524", "10.28.224.185"],
            "server3": ["/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/hanseo/diskbot/keykey.pem", "30640", "10.28.224.203"],
            "server4": ["/data/ephemeral/home/level2-nlp-generationfornlp-nlp-08-lv3/hanseo/diskbot/keykey.pem", "30455", "10.28.224.157"]
        }

        # 이전 디스크 사용량 값을 저장할 디렉토리 설정
        self.data_dir = Path(os.path.expanduser("~")) / "disk_monitor_data"
        self.data_dir.mkdir(exist_ok=True)  # 디렉토리가 없으면 생성

        # 이전 값과 현재 값을 저장할 파일 경로
        self.prev_values_file = self.data_dir / "disk_usage_prev.txt"

        # 초기 실행 여부를 확인하기 위한 플래그 파일
        self.initial_run_file = self.data_dir / "initial_run.txt"

        # 마지막 스트레칭 알림 시간을 저장
        self.last_stretch_time = time.time()

    def check_disk_usage(self):
        """
        각 서버의 디스크 사용량을 확인하고 현재 값을 딕셔너리로 반환
        """
        current_values = {}
        for server_name, server_info in self.servers.items():
            key_path, port, ip = server_info
            try:
                # SSH 명령어 구성
                ssh_command = [
                    "ssh",
                    "-i", key_path,
                    "-p", port,
                    "-o", "StrictHostKeyChecking=no",  # 호스트 키 검증 건너뛰기
                    "-o", "UserKnownHostsFile=/dev/null",  # known_hosts 파일 무시
                    f"root@{ip}",
                    "df -h | grep '/data/ephemeral' | awk '{print $4}'"
                ]
                # SSH 명령어 실행
                result = subprocess.run(
                    ssh_command,
                    capture_output=True,
                    text=True,
                    timeout=self.SSH_TIMEOUT
                )
                if result.returncode == 0:
                    avail = result.stdout.strip()
                    if avail:
                        current_values[server_name] = avail
                else:
                    print(f"{server_name} 에서 오류 발생: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"{server_name} 에 연결 시 타임아웃 발생")
            except Exception as e:
                print(f"{server_name} 체크 중 오류 발생: {e}")
        return current_values

    def send_slack_message(self, message):
        """
        Slack 채널로 메시지를 전송
        """
        try:
            payload = {"text": message}
            response = requests.post(
                self.slack_webhook_url,
                json=payload,
                timeout=self.SLACK_TIMEOUT
            )
            if response.status_code != 200:
                print(f"Slack 메시지 전송 오류: {response.status_code}")
        except Exception as e:
            print(f"Slack 메시지 전송 중 오류 발생: {e}")

    def send_initial_status(self, current_values):
        """
        모든 서버의 초기 디스크 사용량 상태를 Slack으로 전송
        """
        message = "📊 현재 모든 서버의 디스크 사용량 현황\n\n"
        for server_name, value in current_values.items():
            message += f"• {server_name}: {value}\n"
        self.send_slack_message(message)

    def send_stretching_message(self):
        """
        스트레칭 시간 알림 메시지를 Slack으로 전송
        """
        message = "🧘 스트레칭 하실 시간입니다.\nhttps://www.youtube.com/watch?v=AEbr_-Z86tU"
        self.send_slack_message(message)

    def run(self):
        """
        주요 모니터링 로직 실행
        """
        # 현재 디스크 사용량 확인
        current_values = self.check_disk_usage()

        # 초기 실행 여부 확인
        if not self.initial_run_file.exists():
            # 초기 상태 전송
            self.send_initial_status(current_values)
            # 초기 실행 표시
            with open(self.initial_run_file, 'w') as f:
                f.write('done')
            # 현재 값을 이전 값으로 저장
            with open(self.prev_values_file, 'w') as f:
                json.dump(current_values, f)
            return

        # 이전 값 로드
        try:
            with open(self.prev_values_file, 'r') as f:
                prev_values = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"이전 값 읽기 오류: {e}")
            prev_values = {}

        # 변화사항 확인 및 Slack 알림 전송
        for server_name, current_value in current_values.items():
            prev_value = prev_values.get(server_name)
            if current_value != prev_value:
                # 변화량 계산
                try:
                    current_num = float(current_value.rstrip('G'))
                    prev_num = float(prev_value.rstrip('G')) if prev_value else 0
                    change = current_num - prev_num
                    change_str = f"({'증가' if change > 0 else '감소'}: {abs(change):.1f}G)"
                except (ValueError, AttributeError):
                    change_str = ""
                message = (
                    f"💾 디스크 사용량 변동 알림\n"
                    f"• 서버: {server_name}\n"
                    f"• 현재 가용량: {current_value}\n"
                    f"• 이전 가용량: {prev_value if prev_value else '정보 없음'}\n"
                    f"• 변동사항: {change_str}"
                )
                self.send_slack_message(message)

        # 현재 값을 이전 값으로 저장
        with open(self.prev_values_file, 'w') as f:
            json.dump(current_values, f)

        # 스트레칭 메시지 전송 여부 확인
        if time.time() - self.last_stretch_time >= self.STRETCH_INTERVAL:
            self.send_stretching_message()
            self.last_stretch_time = time.time()


def main():
    monitor = DiskMonitor()
    while True:
        try:
            monitor.run()
            print(f"모니터링 실행 완료: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(DiskMonitor.MONITOR_INTERVAL)
        except Exception as e:
            print(f"메인 루프 오류: {e}")
            time.sleep(DiskMonitor.MONITOR_INTERVAL)


if __name__ == "__main__":
    main()


# curl -X POST -H 'Content-type: application/json' \
# --data '{"text":"위 위 위 위플래시"}' \

# rm ~/disk_monitor_data/initial_run.txt
