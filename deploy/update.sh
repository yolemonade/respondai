#!/bin/bash
# 코드 업데이트 시 실행 (로컬에서 HF Spaces에 push한 뒤 VM에서 실행)
set -e
cd ~/respondai
git pull
source venv/bin/activate
pip install -r requirements.txt -q
sudo systemctl restart respondai
echo "업데이트 완료. 상태:"
sudo systemctl status respondai --no-pager
