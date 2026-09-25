# 내 컴퓨터에 설치하기 (로컬 설치 키트) / Local install kit

연구실에서 받은 **설치 키트(ZIP)** 와 **비밀번호**로 STL Simulator를 내 컴퓨터에 설치합니다. 시뮬레이터는
Docker 안에서 실행되고, 브라우저에서 `http://127.0.0.1:8000` 으로 씁니다. 인터넷에 공개되지 않고 **이 컴퓨터에서만**
접속됩니다. 키트 안의 프로그램은 암호화되어 있어서 비밀번호 없이는 설치하거나 내용을 볼 수 없습니다.

> 발표 전 연구 내용입니다. 키트와 비밀번호를 연구실 밖으로 공유하지 마세요.

English summary: unzip the kit, run `install-windows.bat` / `install-mac.command` / `bash install-linux.sh`, enter the
password; the installer checks Docker, builds the simulator image locally (5–15 min the first time), starts it on
`http://127.0.0.1:8000` (or the first free port 8001–8010) and opens the browser. The maintainer section at the end
explains how to build a kit.

---

## 1. 준비물

| | Windows | macOS | Linux |
|---|---|---|---|
| 운영체제 | Windows 10 (22H2) / 11, 64비트 | Docker Desktop이 지원하는 버전 (최근 3개 주요 버전) | Ubuntu, Debian, Fedora 등 64비트 |
| Docker | Docker Desktop (무료) + **WSL2** | Docker Desktop (Intel·Apple Silicon 모두 가능) | **Docker Engine 28 이상** 또는 Docker Desktop (28 미만은 같은 네트워크의 다른 컴퓨터가 시뮬레이터에 접속할 수 있어 설치 파일이 경고하고 멈춤) |
| 메모리 | 8 GB 이상 권장 (Docker에 4 GB 이상) | 8 GB 이상 권장 | 4 GB 이상 |
| 디스크 | C: 드라이브 여유 5 GB 이상 | 여유 5 GB 이상 | `/var/lib/docker` 쪽 여유 5 GB 이상 |
| 인터넷 | 처음 설치할 때 필요 (Docker Hub, npm, PyPI에서 부품을 받아 조립) | 같음 | 같음 |
| 기타 | CPU 가상화(BIOS의 VT-x/SVM) 켜짐 | — | `bash`, `curl` 또는 `wget`. 설치 파일은 **sudo 없이** 실행 |

설치가 끝난 뒤에는 인터넷 없이도 쓸 수 있습니다.

## 2. 설치 순서

### 2-1. Docker Desktop 설치 (처음 한 번)

- <https://www.docker.com/products/docker-desktop/> 에서 받아 설치하고, **한 번 실행해서 약관에 동의**해 둡니다.
- Windows: 설치 중 WSL2 설치를 안내하면 따르고, 재부팅하라고 하면 재부팅합니다.
- Docker Desktop이 없으면 설치 파일이 대신 설치를 도와줍니다 (Windows: `winget`, macOS: Homebrew, Linux: 공식
  `get.docker.com` 스크립트 — 반드시 물어본 뒤에만 설치). 설치가 끝나면 설치 파일을 **다시 실행**하세요.

### 2-2. 키트 압축 풀기

`STL-Simulator-Installer-<버전>.zip` 을 **전부** 풉니다 (Windows: 오른쪽 클릭 → "모두 압축 풀기"). 압축을 풀지 않고
ZIP 안에서 바로 실행하면 "installer files were not found" 안내가 나옵니다.

### 2-3. 설치 파일 실행

| OS | 실행 방법 | 처음 실행할 때 나올 수 있는 경고 |
|---|---|---|
| Windows | `install-windows.bat` 더블클릭 | "Windows의 PC 보호" → **추가 정보 → 실행**. "게시자를 확인할 수 없습니다" → **실행** |
| macOS | `install-mac.command` 더블클릭 (터미널 창이 열림) | "확인되지 않은 개발자" → 시스템 설정 → 개인정보 보호 및 보안 → **그래도 열기**. 또는 터미널에서 `bash ` 입력 후 파일을 끌어다 놓고 Enter. 그래도 막히면 터미널에서 `xattr -dr com.apple.quarantine ` 입력 후 **키트 폴더**를 끌어다 놓고 Enter, 다시 더블클릭. "터미널이 다운로드 폴더에 접근하려고 합니다" → **허용** (거부했다면 시스템 설정 → 개인정보 보호 및 보안 → 파일 및 폴더 → 터미널 → 다운로드 폴더 켜기, 또는 키트 폴더를 데스크탑으로 옮김) |
| Linux | 파일 관리자에서 키트 폴더를 열고 빈 곳 오른쪽 클릭 → "터미널에서 열기" → `bash install-linux.sh` (**sudo 없이**. 필요할 때만 설치 파일이 sudo를 물어봄) | — |

### 2-4. 화면에 나오는 것

```
STL Simulator 설치 (버전 2026.10.01)
==> [1/6] Docker 확인        Docker: x86_64, CPU 8, 메모리 7900 MB → 계산 프로세스 7개
==> [2/6] 설치 위치          설치 폴더 [C:\Users\…\AppData\Local\STL-Simulator]  (Enter = 기본값)
==> [3/6] 비밀번호 확인      비밀번호를 입력하세요 (입력 내용은 보이지 않습니다)
==> [4/6] 시뮬레이터 이미지 만들기   [00:03] [web 3/6] RUN npm ci …   [warmup] … (처음 5–15분)
==> [5/6] 실행               시뮬레이터가 실행 중입니다: http://127.0.0.1:8000
==> [6/6] 마무리             설치 완료! 주소: http://127.0.0.1:8000/
```

1. **Docker 확인**: Docker가 꺼져 있으면 켜고 최대 3분 기다립니다. 계산 프로세스 수는 CPU − 1 (최대 8)이고 메모리가
   적으면 줄입니다.
2. **설치 위치**: Windows `%LOCALAPPDATA%\STL-Simulator`, macOS/Linux `~/STL-Simulator`. Enter를 누르면 기본값,
   다른 경로를 입력하면 그곳에 설치합니다 (비어 있는 폴더여야 합니다).
3. **비밀번호**: 입력하는 글자는 보이지 않습니다. 틀리면 "비밀번호가 맞지 않습니다 (한/영 · Caps Lock 확인)"가 나오고
   3번까지 다시 입력할 수 있습니다. 한글 자모가 입력된 것으로 보이면 "한/영 키를 누른 뒤 다시 입력"하라고 알려 줍니다.
   3번 모두 틀리면 **아무것도 설치하지 않고** 끝납니다. 앞뒤 공백은 무시합니다. 메신저로 받은 비밀번호는 복사해서
   붙여 넣어도 됩니다.
4. **이미지 만들기**: 암호화된 프로그램을 Docker 안에서 풀어 바로 이미지를 만듭니다. 처음에는 부품을 내려받고
   계산 엔진을 미리 컴파일하느라 **5–15분** 걸립니다. 진행 줄이 한동안 멈춰 있어도 정상입니다. 자세한 기록은
   설치 폴더의 `logs\build-…log` 에 남습니다. 빌드가 끝나면 Docker 빌드 캐시에 남은 소스 묶음 사본은 바로 지웁니다.
5. **실행**: 컨테이너 `stl-simulator` 를 `127.0.0.1:8000` 에 띄웁니다 (8000번을 다른 프로그램이 쓰면 8001–8010 중
   빈 번호). 결과 캐시는 Docker 볼륨 `stl-simulator-cache` 에 남습니다.
6. **마무리**: 바탕화면·시작 메뉴 아이콘(Windows), 설치 폴더의 "STL Simulator 열기.command"(macOS), 프로그램 메뉴
   항목(Linux)을 만들고 브라우저를 엽니다.

## 3. 사용하기

- 주소: `http://127.0.0.1:8000` (설치 끝에 나온 번호). **이 컴퓨터에서만** 열립니다.
- 설치 폴더 구성:

  ```
  STL-Simulator/
    bin/          start · stop · open · status · logs · update · rollback · uninstall  (.bat / .command / .sh)
    logs/         설치·빌드 기록 (비밀번호는 기록되지 않음)
    version.txt   버전, 커밋, 날짜, 주소
    README-사용법.txt
    install-state.txt   (설치 프로그램이 쓰는 설정: 포트, 현재/이전 이미지)
  ```
- Docker가 켜질 때 시뮬레이터도 자동으로 시작됩니다 (`stop` 으로 멈춘 경우는 제외). 쓰지 않을 때 메모리를
  아끼려면 `bin/stop` 을 실행하세요. 다시 쓸 때는 아이콘 또는 `bin/open`.
- 계산 프로세스 하나가 메모리 150–250 MB를 씁니다.

## 4. 업데이트 · 되돌리기 · 제거

- **업데이트**: 새 키트를 받으면 압축을 풀고 그 키트의 설치 파일을 실행합니다. 기존 설치를 찾아서 같은 폴더·같은
  포트로 업데이트하고, 바로 전 버전 이미지는 되돌리기용으로 남깁니다 (그보다 오래된 이미지는 지웁니다). 새 버전이
  시작되지 않으면 **자동으로 이전 버전으로 되돌립니다**. 이미 설치된 것과 같은 키트를 다시 실행하면 복구(다시 빌드)가
  됩니다. 설치 폴더에서 `bin/update <새 키트 폴더>` 로도 실행할 수 있습니다.
- **되돌리기**: `bin/rollback` (또는 `bin/update --rollback`). 한 번 더 실행하면 다시 새 버전으로 돌아갑니다.
- **제거**: `bin/uninstall`. 컨테이너, `stl-simulator:*` 이미지, 아이콘, 설치 폴더를 지우고, 결과 캐시 볼륨과
  **Docker 빌드 캐시**를 지울지 물어봅니다 (둘 다 기본값 "예"). 빌드 캐시에는 시뮬레이터 프로그램의 사본이 들어 있으므로
  지우는 것을 권장합니다. 다른 Docker 프로젝트의 빌드 캐시도 함께 지워져 그 프로젝트의 다음 빌드가 조금 느려질 뿐입니다.
  기본 이미지(`python:3.11-slim`, `node:22-slim`)에는 프로그램이 없으므로 남깁니다.

## 5. 문제 해결

| 증상 | 원인과 해결 |
|---|---|
| "Docker가 설치되어 있지 않습니다" | Docker Desktop을 설치하고 한 번 실행한 뒤 설치 파일을 다시 실행합니다. |
| "Docker 엔진이 …초 안에 시작되지 않았습니다" | Docker Desktop을 처음 켜면 약관 동의(Accept) 전에는 엔진이 시작되지 않습니다. 창에 약관·업데이트·오류 안내가 있는지 봅니다 (로그인은 건너뛰어도 됨). 3분이 지나면 더 기다릴지 물어봅니다. |
| Windows: WSL2 관련 오류 | 관리자 PowerShell에서 `wsl --install` (이미 있으면 `wsl --update`) 후 재부팅. |
| Windows: 가상화 꺼짐 | BIOS/UEFI 설정에서 Intel VT-x 또는 AMD SVM을 켭니다. 설치 파일이 감지하면 알려 줍니다. |
| Windows: Access denied / docker-users | 관리자 PowerShell: `net localgroup docker-users "<사용자이름>" /add` → 로그아웃 후 다시 로그인. |
| Windows 컨테이너 모드 | 작업 표시줄 Docker 아이콘 → "Switch to Linux containers" (설치 파일이 바꿀지 물어봅니다). |
| 비밀번호가 맞지 않습니다 | 대소문자·키보드 한/영 상태를 확인합니다 (macOS는 입력 소스가 모든 앱에 공통이라 한글 상태로 남아 있기 쉽습니다). 3번 틀리면 다시 실행하면 됩니다. 키트마다 비밀번호가 다를 수 있습니다. |
| "설치 파일이 손상되었습니다" | 다운로드가 덜 되었거나 파일이 바뀌었습니다. 키트를 다시 받습니다. |
| 포트가 모두 사용 중 | 8000–8010을 다른 프로그램이 씁니다. `install-… --port 8123` (Windows: `install-windows.bat -Port 8123`). |
| 첫 빌드 중 네트워크 오류 (`dial tcp`, `timeout`, `npm ERR! network`, `Could not fetch URL`) | 인터넷이 필요합니다. 회사·학교 프록시는 Docker Desktop → Settings → Resources → Proxies에 넣습니다. Linux Docker Engine은 두 곳: 이미지 받기용 `sudo systemctl edit docker` → `[Service]` `Environment="HTTPS_PROXY=http://프록시:포트"` 후 `sudo systemctl restart docker`, 빌드 안의 npm/pip용 `~/.docker/config.json` 의 `"proxies": {"default": {"httpsProxy": "http://프록시:포트", "noProxy": "localhost,127.0.0.1"}}`. 방화벽이 `docker.io`, `registry.npmjs.org`, `pypi.org`, `files.pythonhosted.org` 를 막지 않는지 확인. |
| `x509` / `certificate` 오류 | 보안 프로그램이 HTTPS를 검사(가로채기)합니다. 가능하면 다른 네트워크(휴대폰 테더링 등)에서 첫 설치를 합니다. 담당자는 아래 7-4의 `STL_BUILD_EXTRA_ARGS` 방법을 쓸 수 있습니다. |
| Apple Silicon (M1–M4) | 문제 없습니다. arm64용으로 직접 빌드합니다 (에뮬레이션 아님). |
| 디스크 공간 부족 (`no space left on device`) | 5 GB 이상 확보합니다. Docker Desktop → Troubleshoot → Clean / Purge data, 또는 쓰지 않는 이미지 삭제. |
| Docker 메모리 부족 / 계산이 느림 | Docker Desktop → Settings → Resources에서 메모리 4 GB 이상 (Windows WSL2: `%UserProfile%\.wslconfig` 의 `memory=`). |
| Linux: `permission denied` (docker.sock) | `sudo usermod -aG docker $USER` → 로그아웃 후 다시 로그인. 설치 파일을 `sudo` 로 실행하지 마세요 (설치 파일이 거부합니다: root 계정에 설치되어 내 메뉴·브라우저에서 쓸 수 없게 됨). |
| Linux: "Docker Engine … 28.0보다 오래된 버전" | 28 미만에서는 `127.0.0.1` 에만 연 포트도 같은 네트워크의 다른 컴퓨터가 접근할 수 있습니다 (Docker 28.0 보안 수정). <https://docs.docker.com/engine/install/> 대로 공식 저장소의 최신 Docker Engine으로 바꿉니다 (배포판 기본 `docker.io` 패키지는 오래된 경우가 많음). 믿을 수 있는 네트워크에서만 "그래도 설치"를 직접 입력해 진행할 수 있습니다. |
| macOS: "installer-files/stl-local.sh 을(를) 읽을 수 없습니다" | 압축을 전부 풀었는지, "터미널이 다운로드 폴더에 접근" 창에서 거부하지 않았는지 확인 (2-3 표). |
| Windows: 검은 창이 잠깐 떴다 사라짐 / "PowerShell could not run the installer" | 회사·학교 정책이나 백신이 PowerShell 스크립트를 막았습니다. 창의 사진과 함께 담당자에게 알려 주세요. |
| 브라우저가 열리지 않음 | 주소 `http://127.0.0.1:8000` 을 직접 입력합니다. |
| 다른 컴퓨터에서 접속하고 싶음 | 로컬 설치는 이 컴퓨터 전용입니다 (127.0.0.1, 주소도 `127.0.0.1`·`localhost` 만 받음). 여럿이 쓰려면 연구실 서버 배포(docs/DEPLOY.md)를 씁니다. |
| 그 밖의 오류 | 화면의 오류 문구와 설치 폴더의 `logs` 폴더를 담당자에게 보냅니다 (비밀번호는 들어 있지 않습니다). |

## 6. 보안 메모

- 키트의 프로그램(`installer-files/stl-simulator.stlenc`)은 비밀번호로 암호화되어 있습니다: 형식 `STLLOC2` —
  PBKDF2-HMAC-SHA256(200만 회) → AES-256-CTR + HMAC-SHA256. 비밀번호가 틀리면 복호화 전에 거부되고 아무것도 풀리지
  않습니다. 키트의 평문 파일(설치 스크립트, 안내문, 범용 복호화 도우미)에는 모델 코드, 커밋 해시, 비밀번호가 없습니다
  (키트를 만들 때 자동 검사).
- **암호화는 소스 저장소가 비공개일 때만 의미가 있습니다.** GitHub 저장소가 Public이면 누구나 코드를 받을 수 있으므로
  키트를 만들기 전에 저장소가 **Private** 인지 확인합니다 (7-1).
- 복호화는 인터넷이 끊긴 일회용 `python:3.11-slim` 컨테이너 안에서만 일어나고, 풀린 소스는 그 컨테이너의 출력에서
  곧바로 `docker build` 로 들어갑니다. 평문 소스가 **일반 파일로** 디스크에 풀리지는 않습니다. 다만 **Docker의 저장소
  안에는** 프로그램이 들어갑니다: 설치된 이미지(설치되어 있는 동안), 그리고 Docker 빌드 캐시(설치 파일이 빌드 직후
  소스 묶음 사본은 지우고, 이미지 층 캐시는 제거할 때 물어보고 지움). Docker Desktop에서는 이것이 `Docker.raw` /
  `ext4.vhdx` 같은 가상 디스크 파일 안에 있습니다.
- 비밀번호: macOS/Linux는 복호화 컨테이너의 표준입력 첫 줄로 넘기고(명령줄·환경변수·파일·로그에 남지 않음), Windows는
  환경변수 이름으로만 넘깁니다(`-e STL_PW`, 명령줄에는 없음). Windows에서는 1–2초 동안 도는 복호화 컨테이너의 설정에
  비밀번호가 보이므로, 그 순간 이 PC의 Docker를 쓸 수 있는 사람(`docker inspect`)은 볼 수 있습니다. 비밀번호에서 나온
  암호 키는 다른 프로그램에 넘기지 않습니다 (AES는 도우미 프로세스 안에서 실행).
- 설치된 **Docker 이미지 안에는 프로그램이 들어 있습니다.** 그 컴퓨터의 관리자(또는 Docker를 쓸 수 있는 사람)는
  이미지에서 코드를 꺼낼 수 있으므로, 공용 PC에는 설치하지 마세요. 암호화는 키트를 드라이브·USB로 **전달하는 동안**을
  보호합니다.
- 시뮬레이터는 `127.0.0.1` 에만 열리고, 주소(Host)가 `127.0.0.1`·`localhost` 가 아닌 요청은 거부합니다
  (`STL_ALLOWED_HOSTS`): 악성 웹페이지가 DNS rebinding으로 이 컴퓨터의 시뮬레이터를 읽는 것을 막습니다. Linux에서는
  Docker Engine 28 이상이어야 같은 네트워크의 다른 컴퓨터가 접근하지 못합니다 (설치 파일이 확인).
- 키트를 가진 사람은 비밀번호를 오프라인으로 무제한 추측해 볼 수 있습니다. 추측하기 쉬운 비밀번호(연구실 이름 + 숫자
  등)는 PC 한 대로 몇 분이면 풀리므로 가벼운 접근만 막습니다. 빌더는 그런 비밀번호를 기본적으로 거부합니다 (7-1).

---

## 7. 관리자용: 새 키트 만들기 / Maintainers: building a kit

### 7-1. 만들기

**먼저 확인**: GitHub 저장소가 **Private** 인지 (Settings → General → Danger Zone → Change repository visibility).
Public이면 키트의 암호화는 아무것도 보호하지 못합니다.

필요한 것: git, Python 3.9 이상(표준 라이브러리만; Linux에서는 libcrypto로 빠르게, 그 밖에는 내장 AES로 MB당 약 1초),
Docker(있으면 설치 파일과 똑같은 방식으로 복호화 검사를 추가로 함).

```bash
git status                                   # 커밋된 내용만 키트에 들어갑니다 (--ref 로 다른 커밋 지정 가능)
python3 scripts/build_local_bundle.py --generate-password     # 강한 비밀번호 제안 (무작위 6글자 단어 3개를 - 로 이은 형태; 매번 다름)
read -rs STL_BUNDLE_PASSWORD && export STL_BUNDLE_PASSWORD      # 비밀번호 입력 (화면에 안 보임, 기록 안 남음)
python3 scripts/build_local_bundle.py                # 버전 기본값: <커밋 날짜 YYYY.MM.DD>-<빌드 ID 앞 7자리>
python3 scripts/build_local_bundle.py --version 2026.10.01     # 버전 직접 지정
unset STL_BUNDLE_PASSWORD
```

비밀번호 규칙: 16자 미만이거나, 연구실·프로그램 이름(`nobel`, `kaist`, `stl`, `simulator`, `lab` …)이 들어 있거나,
"단어 하나 + 숫자" 꼴이면 빌더가 **거부**합니다 (종료 코드 2, 이유와 추측에 걸리는 시간 표시). 위험을 알고도 그대로
쓰려면 `--allow-weak-password` 를 붙입니다 (경고를 출력하고 만듦). 권장: `--generate-password` 가 만든 비밀번호(무작위
음절 단어 3개, 약 56비트)를 메신저로 따로 알려 주기.

Windows PowerShell: `$s = Read-Host -AsSecureString` →
`$env:STL_BUNDLE_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR([Runtime.InteropServices.Marshal]::SecureStringToBSTR($s))`
→ `python scripts\build_local_bundle.py` → `Remove-Item Env:\STL_BUNDLE_PASSWORD`.

결과: `dist-local/STL-Simulator-Installer-<버전>/` 폴더와 같은 이름의 `.zip` (약 5 MB). ZIP은 연구실 구성원만 볼 수 있게
제한한 **Google Drive 링크**, NAS, USB로 전달하고, **비밀번호는 다른 경로로** (메신저·구두) 알려 줍니다. **메일 첨부는
쓰지 마세요**: Gmail과 많은 학교·회사 메일이 `.bat`/`.cmd`/`.ps1` 이 든 ZIP을 보안상 차단합니다. `dist-local/` 은
커밋하지 않습니다. 키트에는 커밋 해시 대신 빌드 ID(`BUILD_ID`, 커밋에서 계산한 불투명한 값)만 들어가고, 빌더가
`dist-local/kit-builds.txt` 에 "빌드 ID 커밋 버전 날짜"를 한 줄씩 남깁니다 (키트 밖).

스크립트가 하는 검사 (하나라도 실패하면 키트를 쓰지 않음):
- 비밀번호는 환경변수 `STL_BUNDLE_PASSWORD` 에서만 읽음. 없거나, 앞뒤 공백이 있거나, 8자 미만이면 거부. 약한
  비밀번호(위 규칙)는 `--allow-weak-password` 없이는 거부.
- 페이로드 = `git archive <ref>` 중 Docker 빌드에 필요한 것(`Dockerfile`, `.dockerignore`, `engine/`, `server/`,
  `scripts/`, `web/`; `server/tests`, `web/e2e`, 문서 제외). Dockerfile의 `COPY` 원본이 모두 들어 있는지 확인.
- 암호화 후 다시 풀어 원본 아카이브와 SHA-256이 같은지 확인, 틀린 비밀번호가 거부되는지 확인.
- 키트의 모든 평문 파일에 모델 관련 문자열(파라미터 이름·값, 엔진 모듈 이름, `GIDL`, `V_LU` 등), 커밋 해시, 비밀번호가
  없는지 확인. ZIP 내용이 폴더와 같은지 확인. `.sh`/`.command` 에서 `$변수` 바로 뒤에 한글 등 비 ASCII 글자가 오면 거부
  (macOS bash 3.2가 잘못 읽음; `${변수}` 로 씀).
- Docker가 있으면 `python:3.11-slim` 컨테이너에서 설치 파일과 같은 방법(비밀번호는 표준입력 첫 줄)으로 복호화해 SHA-256
  확인, 틀린 비밀번호는 종료 코드 3.

### 7-2. 비밀번호 바꾸기 · 버전

- 비밀번호를 바꾸려면 새 비밀번호로 키트를 다시 만들면 됩니다. 이미 나눠 준 키트는 옛 비밀번호로 계속 풀립니다.
- 설치된 컴퓨터는 버전 문자열로 이미지(`stl-simulator:local-<버전>`)를 구분합니다. 새 키트는 새 버전으로 만드세요
  (같은 버전을 다시 설치하면 "복구"로 다룹니다).

### 7-3. 파일 구성

| 파일 | 역할 |
|---|---|
| `scripts/build_local_bundle.py` | 키트 빌더 |
| `deploy/local/install-windows.bat` | Windows 진입점 (ASCII, PowerShell 실행) |
| `deploy/local/install-mac.command`, `install-linux.sh` | macOS·Linux 진입점 → `stl-local.sh` |
| `deploy/local/installer-files/install-windows.ps1` | Windows 설치·관리 (PowerShell 5.1, UTF-8 BOM). 설치 후 `bin\stl-sim.ps1` |
| `deploy/local/installer-files/stl-pipe.cmd` | Windows 바이너리 파이프 (복호화 컨테이너 → `docker build -`) |
| `deploy/local/installer-files/stl-local.sh` | macOS·Linux 설치·관리 (bash 3.2 호환). 설치 후 `bin/stl-sim.sh` |
| `deploy/local/installer-files/stl_payload.py` | 범용 암복호화 도우미 (형식 STLLOC2, AES는 프로세스 안에서). 모델 코드 없음 |
| `deploy/local/설치-안내.txt` | 키트에 들어가는 안내문 (`{VERSION}`, `{DATE}` 치환) |

빌더가 줄바꿈을 정리합니다: `.bat`/`.cmd`/`.ps1` → CRLF (`.ps1` 은 BOM 포함), `.sh`/`.command` → LF + 실행 권한.

### 7-4. 자동화·시험용 옵션

- 비대화식 설치 (비밀번호를 명령줄에 직접 쓰지 않음: 셸 기록·에이전트 기록에 남음):
  `read -rs STL_INSTALL_PASSWORD && export STL_INSTALL_PASSWORD; bash install-linux.sh --yes --port 18800 --no-browser; unset STL_INSTALL_PASSWORD`
  (또는 `--password-stdin`: `printf '%s\n' "$PW" | bash install-linux.sh --yes --password-stdin …`, `PW` 는 `read -rs PW` 로).
  Windows: `$env:STL_INSTALL_PASSWORD`, `-PasswordStdin`, `-Yes`, `-Port`, `-InstallDir`, `-NoBrowser`, `-NoShortcut`.
  Docker 설치, Linux의 오래된 Docker Engine 경고는 `--yes` 로도 자동 승인되지 않습니다. 제거: `--remove-cache`/`--keep-cache`
  (결과 캐시 볼륨), `--remove-build-cache`/`--keep-build-cache` (Docker 빌드 캐시; Windows `-RemoveBuildCache`/`-KeepBuildCache`).
- `STL_BUILD_EXTRA_ARGS`: `docker build` 에 붙일 추가 인수. HTTPS를 검사하는 프록시 뒤에서는 그 프록시의 CA를 넣은
  기본 이미지를 만들어 `--build-context python:3.11-slim=docker-image://<CA 이미지> --build-context
  node:22-slim=docker-image://<CA 이미지>` 로 바꿔 끼울 수 있습니다 (Dockerfile 수정 없음).
- 설치 후 관리 명령: `bin/stl-sim.sh start|stop|restart|open|status|logs|rollback|update [키트]|uninstall`
  (Windows `bin\stl-sim.ps1` 같은 이름, 각 `.bat`).

### 7-5. GPT/Codex에게 맡길 때

1. 비밀번호를 파일·커밋·명령줄 인수·로그에 쓰지 않는다. 오직 환경변수 `STL_BUNDLE_PASSWORD` 로 넘기고 끝나면 지운다.
2. 커밋되지 않은 변경은 키트에 들어가지 않는다. 먼저 커밋(또는 `--ref`)하고 `python3 scripts/build_local_bundle.py
   --version <새 버전>` 을 실행한다.
3. 출력에서 `round trip SHA-256 ok`, `leak check: … none found`, (Docker가 있으면) `docker check: … refused (exit 3)`
   를 확인한다. 실패하면 키트를 배포하지 않는다.
4. `dist-local/` 은 커밋하지 않는다. ZIP만 전달한다.
5. Docker가 있는 Linux라면 임시 HOME에서 시험 설치 (비밀번호는 `read -rs` 로만 받음, 명령줄에 쓰지 않음):
   `T=$(mktemp -d); read -rs STL_INSTALL_PASSWORD && export STL_INSTALL_PASSWORD; HOME=$T bash dist-local/STL-Simulator-Installer-<버전>/install-linux.sh --yes --port 18800 --no-browser; unset STL_INSTALL_PASSWORD`
   → `curl -s http://127.0.0.1:18800/api/health` 가 `"ok":true` → `HOME=$T bash $T/STL-Simulator/bin/uninstall.sh --yes`
   (다른 프로젝트의 빌드 캐시를 지우지 않으려면 `--keep-build-cache`).
6. 설치 스크립트를 고치면 `bash -n`, `shellcheck`, bash 3.2 호환(연관 배열·`${x,,}`·`mapfile`·`sed -i`·`readlink -f`
   금지, 한글 등 비 ASCII 글자 바로 앞의 변수는 `${VAR}` 로), PowerShell 5.1 호환(`&&`, `||`, `??`, 삼항 연산자,
   `-AsPlainText` 금지; 바이너리는 PowerShell 파이프로 넘기지 않음)을 지킨다.
7. 저장소가 Private인지 확인하기 전에는 키트를 만들거나 보내지 않는다. 키트 ZIP은 메일로 보내지 않는다 (Drive 링크/USB).

### 7-6. 설계 요약 / Design notes

- Payload format `STLLOC2`: `magic "STLLOC2\0" | salt 16 | PBKDF2 iterations 4 (BE, ≥ 600000; 2000000 used) | IV 16 |
  AES-256-CTR ciphertext | HMAC-SHA256 tag 32` over everything before it; master = PBKDF2-HMAC-SHA256(NFC(password),
  salt, iterations, 32 bytes), AES key = HMAC(master, "STL enc"), MAC key = HMAC(master, "STL mac"). (STLLOC1, the first
  draft, derived 64 bytes = two independent PBKDF2 blocks; a guess check against the known gzip header needed only the
  first, so an attacker paid half of what the installer paid.) See `stl_payload.py`.
- Decryption runs in a throwaway `python:3.11-slim` container (`--rm -i --network none`): Python's `hashlib`/`hmac` +
  AES-CTR in-process (libcrypto through `ctypes`, checked with the NIST SP 800-38A test vector; built-in AES fallback).
  No key is ever put on a command line (an `openssl enc -K <key>` process would show the key to every user of the
  machine in `ps`/`/proc`, container processes included). The helper script travels base64-encoded in an environment
  variable (`STL_HELPER_B64`). Password: macOS/Linux and the builder send it as the first line of stdin
  (`STL_PW_STDIN=1`, `printf` is a bash builtin); Windows passes `STL_PW` by name (`-e STL_PW`) because cmd.exe cannot
  pipe a Unicode password unchanged — visible via `docker inspect` for the 1–2 s the helper runs. No bind mounts
  (Windows paths with spaces/Korean user names are never handed to Docker).
- The payload goes in on stdin and the decrypted `tar.gz` comes out on stdout straight into `docker build -`, so the
  plaintext source is never unpacked to a regular file on the host. The helper writes nothing unless the tag and the
  expected SHA-256 (from `kit-info.txt`) both match. BuildKit keeps the build context as cache records
  (`http url http://buildkit-session/…` and `copy /context /`); the installers delete exactly those after every build
  (`docker buildx prune -f --filter 'description~=^copy./context./$'` and `…=^http.url.http://buildkit-session/`: "." for
  the spaces, because Windows PowerShell 5.1 mangles native arguments that contain spaces and quotes) and offer
  `docker buildx prune -af` on uninstall.
- Windows PowerShell 5.1 corrupts binary data piped between native commands, so the two binary steps run in
  `stl-pipe.cmd` (cmd.exe pipes and `<` are byte-exact); PowerShell only sets environment variables, starts
  `cmd /d /c .\stl-pipe.cmd verify|build` from the kit folder and reads the text output.
- The container runs with `--restart unless-stopped`, `-p 127.0.0.1:<port>:8000`, `STL_WORKERS = min(CPU − 1, 8)` (also
  capped by Docker's memory), `FORWARDED_ALLOW_IPS=127.0.0.1`, `STL_ALLOWED_HOSTS=127.0.0.1,localhost` (Host allow-list
  in `server/main.py`, against DNS rebinding), volume `stl-simulator-cache:/app/server/.cache`; the password gate
  (`STL_ACCESS_PASSWORD`) stays off. Image labels `org.stl-simulator.{version,build}`; the kit carries an opaque
  `BUILD_ID` instead of the commit hash. On Linux the installer requires Docker Engine ≥ 28 (or Docker Desktop /
  rootless), because older engines let LAN neighbours reach ports published on 127.0.0.1.
