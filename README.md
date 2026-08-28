# 비전 기반 자율 의료 배송 드론과 페이로드 탈착 버티포트 시스템

미션 웨이포인트를 비행하며 YOLO로 요구조자를 탐지하고, 접근하여 구호 물자를
투하한 뒤 복귀한다. 복귀 지점에서는 ArUco 마커 기반 비주얼 서보잉으로
버티포트 중앙에 정밀 착륙하고, 리프트가 올린 새 상자를 그리퍼로 체결한다.

GPS 기반 RTL만으로는 cm 단위 정밀 착륙이 어렵기 때문에,
RTL을 조대 정렬(coarse)로, 비주얼 서보잉을 미세 정렬(fine)로 쓰는
이중 루프 구조를 채택했다.

---

## 시스템 구성

| 항목 | 사양 |
|---|---|
| 비행 컨트롤러 | Pixhawk 6C (PX4) |
| 온보드 컴퓨터 | Jetson Orin Nano Super 8GB |
| 카메라 | ELP-USBGS1200P01 (AR0234 글로벌 셔터, 1920×1200) |
| 짐벌 | 2축 (롤/피치). 카본 파이프로 전방 26cm 돌출 |
| 지상국 | QGroundControl |
| 버티포트 | ESP32 + 리프트 서보 2 + 사출 서보 4 |

Pixhawk의 TELEM2와 Jetson을 UART로 연결하고, MAVProxy로 두 포트에 분배한다.

```
Pixhawk ──UART── Jetson ──┬── 14550 : pymavlink (짐벌 제어)
                          └── 14551 : MAVSDK (비행 제어)
```

버티포트는 라우터 WiFi로 연결하며, Jetson이 HTTP 요청을 보내 상자 사출과 리프트를
제어한다. 휴대폰 웹페이지로 사출할 상자를 미리 선택할 수 있다.

---

## FSM 페이즈

| 페이즈 | 동작 |
|---|---|
| P0 INIT | 미션 비행. 요구조자 락온 대기 |
| P1 SEARCHING | 타겟 로스트. 제자리 호버링 |
| P2 APPROACHING | 요 정렬하며 전진 접근. 짐벌 각도로 도달 판정 |
| P3 DROPPING | 5초 대기 → 그리퍼 개방 → 버티포트에 사출 요청 |
| P4 RTL | 복귀. 복귀 중 ArUco 마커 탐색 |
| P5 VERTIPORT_HOLD | 마커 락온. RTL을 끊고 2단계 정렬 |
| P6 DESCENDING | 수평 보정하며 등속 하강 |
| P7 LANDED | PX4 `land()`에 위임. 접지 확인 |
| P8 VERTIPORT | 리프트 상승 → 그리퍼 체결 → 리프트 하강 |

### P5 2단계 정렬

요 회전은 화면 오차의 좌표계를 함께 회전시켜 병진 제어와 간섭한다.
그래서 위치와 방향을 동시에 맞추지 않고 순차적으로 처리한다.

```
POS  : 0번 마커 중심으로 병진만 (요 = 0)
  ↓ 오차 ≤ 0.08
YAW  : 0→1번 마커 벡터로 방향만 (병진 = 0)
  ↓ 오차 ≤ 8도
HOLD : 미세 보정하며 3초 유지 → P6
```

오차가 0.14를 넘으면 언제든 POS로 되돌아간다.

### P6 하강과 착륙 위임

거리 센서가 없어 접지 판정을 직접 할 수 없다. 대신 마커가 보이는 동안만
offboard로 하강하고, 마커를 잃는 시점에 PX4 `land()`에 넘긴다.
접지와 디스암은 PX4의 착륙 감지기가 처리하므로 고도 정보 없이 성립한다.

검출은 한두 프레임씩 깜빡이므로, 1초 연속으로 끊겨야 진짜 유실로 본다.
유예 구간에서는 수평 보정을 멈추고 수직 하강만 유지한다.

---

## 마커 배치

DICT_4X4_50 마커 두 장을 버티포트에 배치한다.

```
      ← 드론 전방
   ┌───────────┐
   │   ID 1    │   20cm. 방향 기준
   └───────────┘
        20cm 간격
   ┌─────────────┐
   │    ID 0     │   22cm. 위치 기준
   └─────────────┘
```

중심 간 거리는 41cm다. 마커 한 변보다 기준선(baseline)이 길어
단일 마커의 코너로 각도를 재는 것보다 yaw 정밀도가 높다.

카메라가 기체 중심보다 전방 26cm에 있으므로, 마커 0의 중심은
그리퍼가 리프트 위에 오는 착지 지점을 기준으로 배치한다.

---

## 설치

### 1. 시스템 패키지

OpenCV는 apt로 먼저 설치한다. GStreamer와 aruco(contrib)가 포함된
빌드여야 한다.

```bash
sudo apt install -y python3-opencv v4l-utils
```

### 2. PyTorch

JetPack 버전에 맞는 NVIDIA 휠을 별도로 설치한다.
데스크톱용 휠을 설치하면 CUDA가 동작하지 않는다.

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 3. 가상환경과 의존성

시스템 OpenCV를 쓰려면 `--system-site-packages`가 필요하다.

```bash
python3 -m venv --system-site-packages ~/venvs/medifly
source ~/venvs/medifly/bin/activate
pip install -r requirements-jetson.txt
```

### 4. TensorRT 변환 (선택)

엔진 파일은 실제로 사용할 Jetson에서 생성해야 한다.

```bash
python3 scripts/export_yolo.py \
  --model models/medifly_person_down_colab/yolov8n_v2_best.pt \
  --format engine --imgsz 640 --device 0 --half
```

`.engine`이 `.pt`와 같은 위치에 있으면 자동으로 사용된다.
`.pt`를 강제하려면 `--no-prefer-engine`을 붙인다.

---

## 실행

터미널 두 개가 필요하다.

```bash
# 터미널 1 — MAVLink 분배
mavproxy.py --master=/dev/ttyTHS1 --baudrate 921600 \
  --out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14551
```

```bash
# 터미널 2 — 메인 루프
source ~/venvs/medifly/bin/activate
python3 scripts/run_webcam_detector.py \
  --model models/medifly_person_down_colab/yolov8n_v2_best.pt \
  --height-m 0.8 --device 0 --marker-size-m 0.22
```

지상 테스트용 키가 있다. 영상 창에 포커스를 준 뒤 누른다.

| 키 | 동작 |
|---|---|
| `4` | P4로 강제 전환 (ArUco 단계 확인) |
| `8` | P8 버티포트 시퀀스 강제 실행 |
| `q` | 종료 |

`--bench-no-rtl`을 붙이면 실제 RTL 명령을 보내지 않는다. 지상 테스트에 쓴다.

### 정지 이미지 검증

카메라와 비행 컨트롤러 없이 YOLO 동작만 확인할 수 있다.
테스트 이미지는 포함하지 않았으므로, 사람 전신이 나온 이미지를 아무거나 준비해 검증한다.

```bash
python3 scripts/run_image_detector.py \
  --model models/medifly_person_down_colab/yolov8n_v2_best.pt \
  --image /경로/사람이미지.jpg \
  --output outputs/result.jpg \
  --device 0
```

---

## PX4 파라미터

QGroundControl에서 설정한다.

| 파라미터 | 값 | 이유 |
|---|---|---|
| `RTL_RETURN_ALT` | 5 m | 미션 고도와 통일 |
| `RTL_DESCEND_ALT` | 5 m | 하강 없이 서보잉으로 진입 |
| `RTL_MIN_DIST` | 3 m | |
| `MPC_XY_CRUISE` | 4.0 | RTL 오버슈트 억제 |

미션 웨이포인트 고도는 5m, 속도는 1m/s를 기준으로 검증했다.

---

## 설정 항목


`flight_fsm.py`와 `scripts/run_webcam_detector.py`의 `SIGN_*` 상수 5개는
**이 프로젝트의 카메라 장착 방향을 기준으로 실측한 값**이다.

| 상수 | 위치 | 대상 |
|---|---|---|
| `SIGN_LATERAL` | flight_fsm.py | 서보잉 좌우 |
| `SIGN_FORWARD` | flight_fsm.py | 서보잉 전후 |
| `SIGN_YAW` | flight_fsm.py | 서보잉 요 |
| `SIGN_APPROACH_YAW` | flight_fsm.py | P2 접근 요 |
| `SIGN_GIMBAL_PITCH` | run_webcam_detector.py | 짐벌 추적 피치 |

카메라를 다시 장착하거나 짐벌 마운트를 바꾸면 부호가 뒤집힐 수 있다.
**부호가 틀린 상태로 비행하면 드론이 목표에서 멀어지는 방향으로 발산한다.**

`for_indoor_test/verify_servo_direction.py`로 지상에서 검증한 뒤 반영한다.
이 도구는 속도 명령을 계산만 하고 전송하지 않는다(DRY RUN).

### 버티포트 주소

`flight_fsm.py`의 `ESP32_URL`을 환경에 맞게 수정한다.

```python
ESP32_URL = "http://172.20.10.12"   # 또는 "http://vertiport.local"
```

핫스팟을 재시작하면 IP가 바뀔 수 있다. 운용 전에 확인한다.

```bash
curl -s http://vertiport.local/ping
```

### 카메라 초점

렌즈는 M12 수동 초점이다. 운용 고도(1~7m)에 맞춰 미리 고정해야 한다.

```bash
python3 for_indoor_test/focus_tune.py
```

화면의 `sharpness` 값을 보면서 렌즈 경통을 돌린다.
숫자가 최대가 되는 지점이 최적이다.

### 실외 노출

자동 노출은 마커가 화면에 들어올 때마다 밝기를 바꿔 검출을 불안정하게 한다.
따라서 실외에서는 고정한다.

```bash
v4l2-ctl -d /dev/video0 -c auto_exposure=1
v4l2-ctl -d /dev/video0 -c exposure_time_absolute=30
```

되돌리려면 `auto_exposure=3`.

---

## 폴더 구성

```
.
├── flight_fsm.py                  FSM 비행 제어 (MAVSDK)
├── requirements-jetson.txt
├── models/
│   └── medifly_person_down_colab/
│       └── yolov8n_v2_best.pt     YOLOv8n person_down
├── scripts/
│   ├── run_webcam_detector.py     메인 실행. 비전 루프 + FSM 스레드
│   ├── run_image_detector.py      정지 이미지 검증
│   └── export_yolo.py             TensorRT 변환
├── src/medifly_vision/
│   ├── detector.py                YOLO 탐지, 적응형 타일링
│   ├── geometry.py                픽셀 ↔ 지상 좌표 변환
│   ├── usb_camera.py              UVC 카메라, 노출 제어
│   └── vertiport_aruco.py         ArUco 검출, 정렬 가이던스
└── for_indoor_test/
    ├── verify_servo_direction.py  서보잉 부호 검증 (DRY RUN)
    └── focus_tune.py              카메라 초점 조정
```

---
