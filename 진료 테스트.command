#!/bin/bash
# Double-click to open interactive test mode in Terminal
cd "$(dirname "$0")"
source .venv/bin/activate
echo "역할을 고르세요"
echo "  1. 내가 의사 (가상 환자에게 직접 질문)"
echo "  2. 내가 환자 (에이전트의 질문에 답하기)"
read -p "번호 (1/2): " r
if [ "$r" = "2" ]; then python eval/play.py --role patient; exit_code=$?; else
  echo ""
  echo "가상 환자 유형을 고르세요"
  echo "  1. 보통 환자"
  echo "  2. 모호한 환자 (짧고 애매하게 답함)"
  echo "  3. 불안한 환자 (걱정이 많고 말이 김)"
  echo "  4. 증상을 축소하는 환자 (별거 아니라고 함)"
  echo "  5. 기억이 흐린 환자 (시간·약 이름을 잘 모름)"
  echo "  6. 무작위 (끝나고 공개)"
  read -p "번호 (1-6): " p
  case "$p" in 2) P=vague;; 3) P=anxious;; 4) P=minimizer;; 5) P=poor_historian;; 6) P=mixed;; *) P=standard;; esac
  python eval/play.py --persona "$P"
fi
read -p "엔터를 누르면 창이 닫힙니다..."
