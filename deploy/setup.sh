#!/bin/bash
# RespondAI — Oracle Cloud ARM VM 초기 세팅 스크립트
# Ubuntu 22.04 기준. 한 번만 실행하면 됩니다.
set -e

echo "=== [1/5] 시스템 패키지 업데이트 ==="
sudo apt-get update -y
sudo apt-get install -y \
    python3-pip python3-venv python3-dev \
    libsndfile1 ffmpeg git curl \
    build-essential pkg-config

echo "=== [2/5] 코드 클론 (HF Spaces) ==="
cd ~
if [ -d "respondai" ]; then
    echo "이미 클론됨. 최신화..."
    cd respondai && git pull && cd ~
else
    git clone https://huggingface.co/spaces/uuyeong/respondai
fi

echo "=== [3/5] Python 가상환경 생성 & 패키지 설치 ==="
cd ~/respondai
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo "=== [4/5] 방화벽 포트 오픈 (7860) ==="
sudo iptables -I INPUT -p tcp --dport 7860 -j ACCEPT
# 재부팅 후에도 유지
sudo apt-get install -y iptables-persistent
sudo netfilter-persistent save

echo "=== [5/5] systemd 서비스 등록 ==="
sudo cp ~/respondai/deploy/respondai.service /etc/systemd/system/respondai.service
sudo sed -i "s|__HOME__|$HOME|g" /etc/systemd/system/respondai.service
sudo systemctl daemon-reload
sudo systemctl enable respondai
sudo systemctl start respondai

echo ""
echo "=== 완료! ==="
echo "상태 확인: sudo systemctl status respondai"
echo "로그 확인: sudo journalctl -u respondai -f"
