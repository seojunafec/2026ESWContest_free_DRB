/*
 * MEDIFLY 버티포트 ESP32
 *
 * 하드웨어
 *   리프트 : MG996R 2개 (GPIO 27, 25). 랙기어 2개를 미러 구동
 *   사출   : 무한회전 서보 4개 (GPIO 18, 19, 21, 22)
 *
 *
 * API
 *   GET /              제어 웹페이지 (휴대폰)
 *   GET /status        상태 JSON
 *   GET /select?box=N  상자 선택 (1~4). 잠금 상태면 거부
 *   GET /eject         선택 상자 사출 + 선택 잠금. 중복 요청은 무시
 *   GET /liftup        리프트 상승
 *   GET /liftdown      리프트 하강
 *   GET /reset         잠금 해제 (수동 복구용)
 *   GET /ping          생존 확인
 *
 *   투하 성공 -> 젯슨 /eject    -> 선택 잠금 + 사출 (리프트에 안착)
 *   RTL -> 착륙 -> 젯슨 /liftup  -> 리프트 상승
 *               -> 젯슨이 그리퍼 체결
 *               -> 젯슨 /liftdown -> 리프트 하강
 */

#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <ESP32Servo.h>

// ══════════ 네트워크 ══════════
const char* WIFI_SSID = "medifly_afec";   // 핫스팟 이름
const char* WIFI_PASS = "********";       // 핫스팟 비밀번호 기입
const char* HOSTNAME  = "vertiport";        // http://vertiport.local

// ══════════ 리프트 (MG996R 2개, 미러 구동) ══════════
const int SERVO1_PIN = 27;
const int SERVO2_PIN = 25;
const int PULSE_MIN_US = 500;
const int PULSE_MAX_US = 2400;
const int LIFT_DOWN_US = 500;
const int LIFT_UP_US   = 2188;
const int MIRROR_SUM_US = LIFT_DOWN_US + LIFT_UP_US;
const int LIFT_STEP_US  = 32;               // 한 스텝 펄스폭 증가량
const int LIFT_STEP_MS  = 10;               // 스텝 간 대기. 느릴수록 부드럽다

// ══════════ 사출 (무한회전 서보 4개) ══════════
const int EJECT_PIN[4] = {18, 19, 21, 22};
const int OPEN_US    = 1300;                // 열림 방향 회전
const int CLOSE_US   = 1700;                // 래치 복귀 방향 회전
const int NEUTRAL_US = 1500;                // 정지
const int OPEN_MS[4]  = {577, 536, 536, 577};   // 실측값
const int CLOSE_MS[4] = {660, 569, 569, 635};   // 실측값
const int DROP_WAIT_MS = 800;               // 상자 낙하 대기 후 래치 복귀

// ══════════ 안전 ══════════
const unsigned long LIFT_TIMEOUT_MS = 25000; // 하강 명령이 없으면 자동 하강

Servo servo1, servo2;
Servo ejectServo[4];
WebServer server(80);

int  selectedBox = 1;        // 1~4
bool locked      = false;    // 사출 후 선택 잠금
bool liftIsUp    = false;
unsigned long liftUpAt = 0;
String lastEvent = "부팅";

// ────────────────────────────────────────────────
// 리프트 구동
// ────────────────────────────────────────────────
void liftWriteRaw(int us) {
  us = constrain(us, LIFT_DOWN_US, LIFT_UP_US);
  servo1.writeMicroseconds(us);
  servo2.writeMicroseconds(MIRROR_SUM_US - us);   // 반대 방향
}

void liftRaise() {
  for (int us = LIFT_DOWN_US; us < LIFT_UP_US; us += LIFT_STEP_US) {
    liftWriteRaw(us);
    delay(LIFT_STEP_MS);
  }
  liftWriteRaw(LIFT_UP_US);
  liftIsUp = true;
  liftUpAt = millis();
}

void liftLower() {
  for (int us = LIFT_UP_US; us > LIFT_DOWN_US; us -= LIFT_STEP_US) {
    liftWriteRaw(us);
    delay(LIFT_STEP_MS);
  }
  liftWriteRaw(LIFT_DOWN_US);
  liftIsUp = false;
}

// ────────────────────────────────────────────────
// 사출 구동
// ────────────────────────────────────────────────
void spin(int i, int us, int ms) {
  ejectServo[i].writeMicroseconds(us);
  delay(ms);
  ejectServo[i].writeMicroseconds(NEUTRAL_US);
  delay(200);
}

void ejectBox(int box) {                     // box: 1~4
  int i = box - 1;
  ejectServo[i].setPeriodHertz(50);
  ejectServo[i].attach(EJECT_PIN[i], PULSE_MIN_US, PULSE_MAX_US);
  ejectServo[i].writeMicroseconds(NEUTRAL_US);
  delay(200);
  spin(i, OPEN_US,  OPEN_MS[i]);             // 래치 열림
  delay(DROP_WAIT_MS);                       // 상자 낙하 대기
  spin(i, CLOSE_US, CLOSE_MS[i]);            // 래치 복귀
  ejectServo[i].detach();                    // 대기 전류 차단
}

// ────────────────────────────────────────────────
// 웹페이지
// ────────────────────────────────────────────────
const char PAGE[] PROGMEM = R"HTML(
<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vertiport</title><style>
body{font-family:-apple-system,sans-serif;background:#111;color:#eee;
     margin:0;padding:20px;text-align:center}
h2{font-weight:500;margin:10px 0 20px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:340px;margin:0 auto}
.b{padding:28px 0;font-size:22px;border:2px solid #444;border-radius:12px;
   background:#1c1c1c;color:#eee}
.b.sel{border-color:#4a9eff;background:#12314f;color:#fff}
.b:disabled{opacity:.35}
#st{margin:22px 0;font-size:15px;color:#8a8a8a;min-height:22px}
.man{margin-top:28px;border-top:1px solid #333;padding-top:16px}
.mb{padding:12px 18px;margin:4px;font-size:14px;border:1px solid #444;
    border-radius:8px;background:#1c1c1c;color:#bbb}
</style></head><body>
<h2>상자 선택</h2>
<div class="grid">
<button class="b" id="b1" onclick="sel(1)">1</button>
<button class="b" id="b2" onclick="sel(2)">2</button>
<button class="b" id="b3" onclick="sel(3)">3</button>
<button class="b" id="b4" onclick="sel(4)">4</button>
</div>
<div id="st">연결 중...</div>
<div class="man">
<button class="mb" onclick="go('/eject')">수동 사출</button>
<button class="mb" onclick="go('/liftup')">리프트 UP</button>
<button class="mb" onclick="go('/liftdown')">리프트 DOWN</button><br>
<button class="mb" onclick="go('/reset')">잠금 해제</button>
</div>
<script>
function sel(n){fetch('/select?box='+n).then(poll)}
function go(u){fetch(u).then(poll)}
function poll(){fetch('/status').then(r=>r.json()).then(s=>{
  for(let i=1;i<=4;i++){
    let b=document.getElementById('b'+i);
    b.className=(i==s.box)?'b sel':'b';
    b.disabled=s.locked;
  }
  document.getElementById('st').textContent=
    (s.locked?'잠금 · ':'대기 · ')+s.box+'번 · '+s.lift+' · '+s.event;
}).catch(()=>{document.getElementById('st').textContent='연결 끊김'})}
setInterval(poll,1000);poll();
</script></body></html>
)HTML";

// ────────────────────────────────────────────────
// 핸들러
// ────────────────────────────────────────────────
void handleRoot() { server.send_P(200, "text/html", PAGE); }

void handlePing() { server.send(200, "application/json", "{\"ok\":true}"); }

void handleStatus() {
  String j = "{\"box\":" + String(selectedBox) +
             ",\"locked\":" + String(locked ? "true" : "false") +
             ",\"lift\":\"" + String(liftIsUp ? "up" : "down") +
             "\",\"event\":\"" + lastEvent + "\"}";
  server.send(200, "application/json", j);
}

void handleSelect() {
  if (locked) { server.send(409, "text/plain", "LOCKED"); return; }
  int b = server.arg("box").toInt();
  if (b < 1 || b > 4) { server.send(400, "text/plain", "BAD_BOX"); return; }
  selectedBox = b;
  lastEvent = String(b) + "번 선택";
  Serial.printf("선택: %d번\n", b);
  server.send(200, "text/plain", "OK");
}

void handleEject() {
  // 젯슨의 esp() 는 실패 시 재시도한다.
  // 사출은 끝났는데 응답만 유실되면 상자가 두 개 나가므로,
  // 이미 잠긴 상태의 재요청은 실행하지 않고 성공으로만 답한다.
  if (locked) {
    Serial.println("!!! 중복 사출 요청 무시");
    server.send(200, "application/json",
                "{\"ok\":true,\"box\":" + String(selectedBox) +
                ",\"repeat\":true}");
    return;
  }

  int b = selectedBox;
  locked = true;
  lastEvent = String(b) + "번 사출";
  Serial.printf(">>> 사출: %d번 (GPIO %d)\n", b, EJECT_PIN[b - 1]);
  ejectBox(b);
  Serial.println("<<< 사출 완료");
  server.send(200, "application/json",
              "{\"ok\":true,\"box\":" + String(b) + "}");
}

void handleLiftUp() {
  if (liftIsUp) {
    liftUpAt = millis();                     // 타임아웃 연장
    server.send(200, "application/json", "{\"ok\":true,\"lift\":\"up\"}");
    return;
  }
  lastEvent = "리프트 상승";
  Serial.println(">>> 리프트 상승");
  liftRaise();
  Serial.println("<<< 상승 완료");
  server.send(200, "application/json", "{\"ok\":true,\"lift\":\"up\"}");
}

void handleLiftDown() {
  lastEvent = "리프트 하강";
  Serial.println(">>> 리프트 하강");
  liftLower();
  Serial.println("<<< 하강 완료");
  server.send(200, "application/json", "{\"ok\":true,\"lift\":\"down\"}");
}

void handleReset() {
  locked = false;
  lastEvent = "잠금 해제";
  Serial.println("잠금 해제");
  server.send(200, "text/plain", "OK");
}

// ────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n>>> MEDIFLY VERTIPORT ESP32 <<<");

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  servo1.setPeriodHertz(50);
  servo1.attach(SERVO1_PIN, PULSE_MIN_US, PULSE_MAX_US);
  servo2.setPeriodHertz(50);
  servo2.attach(SERVO2_PIN, PULSE_MIN_US, PULSE_MAX_US);
  liftWriteRaw(LIFT_DOWN_US);
  Serial.println("리프트 하단 홀드");

  WiFi.mode(WIFI_STA);
  WiFi.setHostname(HOSTNAME);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  WiFi.setSleep(false);            // 절전 해제. 응답 지연을 크게 줄인다

  Serial.print("WiFi 접속");
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 60) {
    delay(500); Serial.print("."); tries++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print(">>> IP 주소: ");
    Serial.println(WiFi.localIP());
    Serial.printf(">>> http://%s.local/\n", HOSTNAME);
    Serial.printf(">>> RSSI: %d dBm\n", WiFi.RSSI());
    if (MDNS.begin(HOSTNAME)) MDNS.addService("http", "tcp", 80);
  } else {
    Serial.println("!!! WiFi 접속 실패. 서보 제어는 계속 가능하다.");
  }

  server.on("/",         handleRoot);
  server.on("/ping",     handlePing);
  server.on("/status",   handleStatus);
  server.on("/select",   handleSelect);
  server.on("/eject",    handleEject);
  server.on("/liftup",   handleLiftUp);
  server.on("/liftdown", handleLiftDown);
  server.on("/reset",    handleReset);
  server.begin();
  Serial.println("서버 시작. 대기 중");
}

void loop() {
  server.handleClient();

  // 리프트 상승 후 하강 명령이 안 오면 자동 하강
  if (liftIsUp && millis() - liftUpAt > LIFT_TIMEOUT_MS) {
    Serial.println("!!! 타임아웃 - 리프트 자동 하강");
    lastEvent = "타임아웃 하강";
    liftLower();
  }

  // WiFi 끊기면 재접속
  static unsigned long lastCheck = 0;
  if (millis() - lastCheck > 5000) {
    lastCheck = millis();
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi 끊김 - 재접속");
      WiFi.disconnect();
      WiFi.begin(WIFI_SSID, WIFI_PASS);
    }
  }
}
