# SOP-Guard（核心引擎 v3：物件錨點 ＋ 手勢 ＋ 清單模式）

規格書（不在本倉庫；第 3、4 節的「固定 ROI、依序狀態機」是 schema 1 的舊設計，仍可用；本資料夾已進到 schema 2，差異見下）。

## 架構（為什麼不是固定框）

| 層 | 做法 | 檔案 |
| --- | --- | --- |
| 手部 | MediaPipe HandLandmarker 逐格，21 點 | `core_detector.py: HandTracker` |
| 物件錨點 | YOLO-World 零訓練，給文字（"cardboard box"、"sneaker"）每 N 格找一次；沒新結果沿用上次（`max_age_sec`） | `object_detector.py`、`scene.py: SceneTracker` |
| 區域 | 相對錨點框的子矩形（`rel`），或固定矩形（舊式 roi） | `scene.py: ZoneSpec / resolve_zone` |
| 手勢 | 1 秒視窗的 21 點軌跡 ＋ 手相對鞋子／鞋盒位置 → RandomForest | `gesture_classifier.py` |
| 步驟條件 | `hand_dwell`（手在區域）、`object_in_zone`（物件進區域）、`gesture`（手勢機率） | `core_detector.py: SOPStateMachine._condition` |
| 狀態機 | **依序模式**（舊；跳步／超時）或 **清單模式**（一輪用 `cycle_end` 事件結束，輪內步驟不分先後各自累計，結束時缺的一次報出） | `core_detector.py: _update_ordered / _update_checklist` |

實測結論（`包裝模擬影片鞋盒外盒.mp4`，實際 4 輪；阿丘科技 demo 的輸出是「正确周期 4、错误周期 1」）：

| 設定 | 結果 |
| --- | --- |
| 固定 ROI＋依序（schema 1，`範例_外盒`） | 5 輪、9 次跳步誤報 |
| 錨點＋依序，A 單獨（`範例_外盒_v2a`） | 7 輪、8 次誤報 |
| 錨點＋依序，A＋B（`範例_外盒_v2`） | 1 輪、11 次誤報（操作者鞋↔盒來回，「依序」前提不成立） |
| **錨點＋清單，A 單獨（`範例_外盒_v3a`）** | 5 輪完整、0 誤報 |
| **錨點＋清單，A＋B（`範例_外盒_v3`）** | **4 輪完整＋1 輪「漏做：確認尺碼」（26.0 秒）**；阿丘的標籤裡第一輪也沒有尺码确认，所以這一筆是對的 |

`包裝模擬影片鞋盒內盒.mp4`（實際 6 輪：開盒 → 拿鞋 → 放入盒身 → 蓋上拿走）：

| 設定 | 結果 |
| --- | --- |
| 錨點＋依序（`範例_內盒_v2`） | 4 輪、1 次超時 |
| 錨點＋清單，必做「鞋子在盒內」（`範例_內盒_v3`） | 4 輪完整、2 輪漏做（黑鞋在深色盒內 YOLO-World 沒抓到）、2 段短於 10 秒的假輪次丟掉 |
| **錨點＋清單，必做「手伸進盒身 1 秒」、「鞋子在盒內」選配（`範例_內盒_v3b`）** | **6 輪完整、0 誤報、2 段丟掉** |

內盒學到的三件事：鞋盒打開時是「盒蓋」「盒身」兩個框，錨點要用 `pick: "bottommost"` 不能用 `largest`；「東西放進去」用手當代理（`hand_dwell` 在 `box_inner`）比靠偵測到盒內的物件穩；`cycle_min_sec` 擋掉「拿起東西走出畫面」造成的假輪次。

手勢分類器留一輪交叉驗證：整體 73%（764 個 1 秒視窗）；`尺码确认` 召回 0.92、`鞋垫` 0.72、`等待` 0.83、`右鞋` 0.60（只有 25 個樣本）、`标签确认` 0.00（阿丘這個標籤靠的是看到標籤，不是手的動作，學不到——確認標籤改用「手在鞋盒區」）。**注意：`範例_外盒_v3` 用的模型是同一支影片訓的，上表 A＋B 的結果偏樂觀；73% 才是對未見過資料的估計。**

## 安裝

```powershell
python -m venv C:\sopg\venv                       # 路徑要短
C:\sopg\venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
C:\sopg\venv\Scripts\python.exe -m pip install -r requirements.txt
C:\sopg\venv\Scripts\python.exe -m pip install "git+https://github.com/ultralytics/CLIP.git"
# 模型：models\hand_landmarker.task（見 models\README.md）、models\yolov8s-worldv2.pt
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt" -OutFile "models\yolov8s-worldv2.pt"
```

建議 venv：專案資料夾下的 `.venv`（含 torch 2.14.1 CPU、ultralytics 8.3.40、mediapipe 0.10.14、scikit-learn 1.5.2）。`start_ui.bat`／`start_web.bat` 預設用 `.venv\Scripts\python.exe`，可用環境變數 `SOPG_PY` 改指別的 python。

## 桌面即時測試（第 1 批工具）

```powershell
cd <專案資料夾>
$env:PYTHONUTF8='1'
$PY=".venv\Scripts\python.exe"
$W="models\yolov8s-worldv2.pt"

# a. 先看相機裡的東西 YOLO-World 認不認得（調 prompt 與 min_conf 用；按 q 離開）
& $PY tools\anchor_probe.py --source 0 --prompts "cardboard box,sneaker" --yolo $W

# b. 從相機建專案骨架（讀實際解析度與實測 fps；第一個錨點是容器、第二個是物品）
& $PY tools\new_project.py 桌面測試 --camera 0 --anchors "box=cardboard box,item=sneaker"

# c. 即時跑（相機來源 YOLO 走背景執行緒，畫面不卡；--stats-log 每分鐘記 FPS／記憶體，掛整晚用）
& $PY core_detector.py --project sop_projects\桌面測試 --source 0 --yolo $W --show --stats-log cache\stats_cam.csv
```

- 相機 fps 以實測為準（這台 webcam 回報 0、實測 15～24，隨光線自動曝光變動）。
- `--max-seconds 28800` 可以限定跑 8 小時後自動停，搭配 `--stats-log` 看 `rss_mb` 有沒有一直漲。
- 影片檔來源仍是同步偵測（每 `detect_every_n_frames` 格一次），結果才能重現；`--sync-yolo` 可強制相機也同步。

## 操作介面（第 3 批）：`python main_ui.py`

操作者只做三步，不碰 JSON：

1. **① 錄製示範**：開啟相機 → 開始錄製 → 把整套動作做一遍 → 停止（存成專案的 `demo.avi`）。也可「匯入影片」。
2. **② 分段說明動作**：拉時間軸 → 設為起點 → 設為終點 → 輸入動作名稱 → 加入這一段；一段一段標完，沒標到的自動當「其他」。按「訓練並產生步驟」：抽特徵（有進度條）→ 依片段分組交叉驗證 → 存模型 → 每個動作寫成一個必做步驟（順序照示範時第一次出現）。
3. **③ 監控**：啟動監控（相機，或勾「用示範影片測試」）、重置／解除警告、輪數統計、右側事件清單。

**自動找其他輪**（分頁②的按鈕）：先把第一輪標好（每個動作各一段、照順序），按「自動找其他輪」，系統用這一輪訓一個草稿模型掃影片其餘部分，再用 Viterbi 依「同樣順序循環」解碼，提出淡色的「建議」分段（表格「狀態」欄寫「建議 NN%」）。對的按「接受選取的建議」或「接受全部建議」，接受後才算已標；接受一輪後再按一次會更準。只標第一輪時建議的準確度有限，**一定要人看過**。用 6 輪鞋盒影片（24 段）實測，以「與真實分段重疊 ≥ 50%」算對：只標第 1 輪 → 建議 18 段，對 15／20、多餘或認錯 3；標 2 輪 → 建議 14 段，對 13／16、多餘 1；標 3 輪 → 建議 11 段，對 9／12、多餘 2（`prob_min` 0.3；沒有順序解碼前只有 7／20）。

一輪怎麼算結束（由上到下優先）：分段裡標了「結束」（或「完成」）的手勢 → 用那個手勢（做完一輪對鏡頭比 OK 之類）；config 已有物件型 `cycle_end`（例如鞋子離開）→ 保留；都沒有 → 「手離開畫面 3 秒」。外盒影片實測：操作者的手幾乎不離開畫面，「手離開」當邊界一輪都結束不了（第二輪三個手勢步驟都達標但永遠等不到結束），所以操作者一定要標「結束」手勢，或由工程師在 `config.json` 設物件型條件。新專案預設 `cycle_min_sec` 5 秒、沒有物件錨點。

```powershell
cd <專案資料夾>
$env:PYTHONUTF8='1'
.venv\Scripts\python.exe main_ui.py
```

- 模型檔固定讀 `models\hand_landmarker.task` 與 `models\yolov8s-worldv2.pt`（後者只有專案設了 anchors 才需要）。
- 監控畫面的中文是 PIL 畫在影格上（`core_detector._draw_dev_overlay`），不是規格書要求的 QPainter；功能相同，之後要換再換。
- 用外盒影片走一次操作者流程（13 段分段、沒有物件錨點、純手勢步驟）：交叉驗證 67%（尺码确认 0.83、鞋垫 0.62、标签确认 0.00）。比有錨點的版本低，因為少了「手相對鞋子／鞋盒」的位置資訊——桌面測試時若準確率不夠，工程師可在 config 加 anchors 與 gesture_anchors 再訓練。
- 離屏冒煙測試 `tests/test_ui_smoke.py` 會把三個分頁的截圖存到 `cache\ui_shots\`。
- **自動走查** `tools\e2e_walkthrough.py`：用「假相機」（播一支影片）把 ①錄製 → ②分段 → 訓練 → ③監控 整條走一遍，每步截圖、收事件、寫 `cache\e2e\report.txt`；出現警告 3 秒後自動按「重置／解除警告」模擬操作者（`--reset-after`），`--monitor-only` 可只重跑監控。這驗的是介面與流程，不是真人手勢的準確率。
  ```powershell
  & $PY tools\e2e_walkthrough.py --video sop_projects\範例_外盒\demo.mp4 --labels cache\outer_labels.csv --monitor-seconds 150
  ```
  外盒影片走查結果：錄製 4558 格（155 MB，假相機播完被介面當「相機斷線」，錄到的檔案照樣收尾保存——這個斷線保護就是走查時發現缺的）；19 段 → 交叉驗證 62%（尺码确认 0.83、鞋垫 0.68、「結束」用手閒著的片段充當 0.66、标签确认 0.09）；監控 13～22 fps。

## 盒內狀態辨識（2026-10-22）：看「盒子裡有什麼」，不看手

使用者指出手勢路線在「判斷物品有沒有放進去」上不可靠（用丟的也該算），改成直接看結果：

- 原理：用 YOLO-World 找盒子（錨點 `box`），裁盒內影像，跟操作者標的狀態範例（空盒／已放卡片／已放鞋子／未開／蓋上…每種 5～10 格）做 CLIP ViT-B/32 向量最近鄰比對（沒有 clip 就退回 32×32 灰階縮圖）；每 5 格判一次、最近 5 次投票 ≥ 3 次才換狀態。免訓練。
- 判不判的規則（`BoxStateRunner`，2026-10-22 用 209 秒內盒影片離線模擬定下來的）：
  - 區域：錨點有多個框時取「最內層」（`state_region`：包住別的候選框的不要）——打開的盒子常同時偵測到「整盒含盒蓋」和「盒身」，範例是盒身裁的，監控黏到整盒就全錯。
  - 手：同一隻手 ≥ 2 個關鍵點落在區域內縮 15% 的範圍才算「手在盒內」不判；手扶盒緣、蓋蓋子照判（`hands_in_inner`）。
  - 只在盒子「1 秒內真的偵測到」時判（`SceneTracker.age`）；錨點沿用中的舊框不判——盒子拿走後舊框對著桌面會判成「空盒」，下一輪會提早開始、然後被新盒子的「未開」結束成假的漏做。
  - 盒子不見（錨點過期）→ 狀態清成「不知道」，不維持上一次——上一輪的「蓋上」留著會被下一輪當結束條件。
  - 盒子剛換位置（被推走／剛放上來）0.5 秒內不判（`STATE_SETTLE_SEC`）：移動中、半出畫面的閉合盒會被判成「空盒」。偵測在「整盒含蓋」「盒身」間切換（一框包住另一框）不算移動。
  - 一輪結束預設＝「盒子被拿走」（`cycle_end` 型別 `object_absent`，錨點消失 2 秒），不是「盒內回到未開」：實測蓋上後 1～1.5 秒就被推走，「未開」撐不到 2 秒，第 1、2 輪會被併成一輪。狀態取名「蓋上／結束／完成」才會改用那個狀態當結束。
  - 範例要挑「手已離開、東西看得到」的格；蓋著的盒子「開盒前」和「蓋上後」都要取幾格當同一個狀態（未開），不然蓋著的盒子會被硬選成別的狀態（實測被判成「已放鞋子」）。
  - `core_detector.py --state-log x.csv` 每次判定寫一列（frame／why＝judged|hands|stale|moving|box_gone／raw／conf／margin／current），調範例或規則先看這個，不要猜。
- 檔案：`state_recognizer.py`（範例庫 `<專案>/states/`、`Embedder`、`StateLibrary`、`StateStream`）、`core_detector.py`（`BoxStateRunner`、`state_region`、步驟型別 `box_state`）、`training.py`（`ensure_state_anchor`、`apply_state_steps`）。
- config：`state_anchor`（錨點名）、`state_rel`（盒內相對區域）、`state_fixed`（鎖定的固定區域）、`state_every_n_frames`；步驟 `{"type": "box_state", "label": "已放卡片"}`；`cycle_end` 可用 `box_state`。
- 介面②：「這一格的盒內狀態是…」→「加入這一格的狀態」（自動找盒子、裁盒內、存範例，顯示範例互驗%）→「用狀態產生步驟」（第一個狀態是起始、名為 結束／完成／蓋上 的狀態當一輪結束，其餘各一個必做步驟）；「鎖定盒子區域」給固定工位；「刪除選取的狀態」。
- 可行性（內盒影片，固定區域，留一輪驗證，49 格）：CLIP 88%、灰階縮圖 82%，「空盒 vs 已放防黴片」只差一顆白色小貼紙仍 6/8 對。
- 整條流程（內盒影片 6 輪，第 1～3 輪取範例、全片監控，步驟 空盒→已放防黴片→已放鞋子、一輪結束＝盒內變回「未開」2 秒）：
  - 第一版（10 格判一次、任一手點在區域就不判、區域黏住、沿用框也判）：209 秒只做完 2 步、0 輪。
  - 離線模擬（`tools/dev/state_trace_record.py` 先把 YOLO／手部結果錄成 `cache/state_trace.json`，`tools/dev/state_sim*.py` 再用快取的 CLIP 向量試規則，一個變體幾秒；範例格用 `state_windows.py` 的 contact sheet 人工挑）：**6 輪全對、0 誤報**；拿掉任一條都退步——只在沿用框上也判 → 5 次假漏做；不清狀態 → 1 次；10 格判一次 → 漏 1 輪；投票改 2/3 → 2 次；結束改 1 秒 → 1 次。
  - 真實跑（`core_detector.py` 整支影片，手部每格偵測）：第一次 5 輪／第 1、2 輪被併成一輪（蓋上後 1.2 秒就推走，「未開」撐不到 2 秒；推走中的盒子還被判成「空盒」）→ 改成「盒子拿走」當結束＋移動中不判 → **5 輪全對、0 誤報、0 丟棄**（第 6 輪影片在盒子拿走前就結束）。模擬比真實樂觀的地方：模擬的手部每 5 格才偵測一次、手扶盒緣的格比較少被擋。
  - 已知風險：剛放上來手還在盒上的蓋著盒子（conf 0.7、margin 0.1～0.19）、掀蓋過程中的盒蓋內側（conf 0.9）會被判成「已放鞋子」約 0.5～1 秒；清單模式只要求一輪內看過，所以不影響計數，但會把「這一輪真的沒放鞋」遮掉。原因是「已放鞋子」範例裡有一格（3035）盒蓋半蓋著——範例要挑鞋子整個看得到、沒有盒蓋入鏡的格；手在蓋著盒子上的格也多標幾格「未開」。

## 物件版（2026-10-27）：框一次、取名、本機訓練小模型，監控時逐樣物品檢查

使用者裁示改走物件版（「先抓出畫面上的物品來命名，設定好後在本機做小模型的訓練」）。免訓練的逐物品偵測在桌燈場景試了四種都不可用
（見 `tools/dev/item_*probe*.py` 與記憶），所以物件版是**輕量訓練**：

- 流程（介面②「標記物品」，使用者嫌按鈕太多後改成三步，一步只看到該步要按的；QStackedWidget＋右下角上一步／下一步）：
  - 第 1 步 標記：每一輪做兩件事——拉到盒子放好、還沒放東西的那一格 →「這一格是空盒」（延伸時的空盒參考）；拉到東西放好的那一格，
    在畫面上拖拉框住物品、輸入名稱 →「加入框選的物品」，每樣物品每一輪都框一次。至少框一樣才能「下一步」。
  - 第 2 步 訓練：一顆「開始訓練」——先整支影片找盒子＋手（`items/trace.json`，有快取），再把每個框選在**同一次擺放**（盒子沒移動）裡
    用樣板比對延伸到其他格（含手在盒內的格）；樣板在空盒參考上也配得到的不算（大塊深色的鞋子會配到陰影，門檻：分數 ≥ 0.75 且比空盒高 0.12）。
    寫成 YOLO 資料集 `cache/item_ds_<專案雜湊>/`（最後一次擺放當驗證）——**不能放在專案資料夾**：ultralytics／`cv2.imread` 在 Windows
    讀不了中文路徑。接著自動開子程序跑 `tools/train_items.py`（yolov8n 微調、CPU、30 epoch、imgsz 640；134 張約 35 分鐘），
    模型 `--copy-to` 回 `items/model/best.pt`；numpy 2.x 沒有 `trapz`，腳本有 shim。模型存在才能「下一步」。
  - 第 3 步 完成：「產生步驟，前往監控」——每樣物品一個 `item` 步驟（一輪內看到過就算），一輪結束＝盒子被拿走 2 秒，並切到分頁③。
  - 舊的動作分段／盒內狀態在「改用手勢／盒內狀態模式（進階）」勾選框裡，預設收起。**三種模式二選一**（使用者裁示）：
    `apply_item_steps` 清掉 gesture／box_state 步驟並把 `gesture_model` 設 None；`apply_gesture_steps` 清掉 item／box_state 步驟；
    `BoxStateRunner` 只在專案有 item 步驟時載入物品模型、只在有 box_state 步驟時載入狀態範例。
- 監控：`BoxStateRunner` 在同一張盒內裁圖上跑模型（每 5 格、手不在盒內、盒子新鮮且停好），`FrameContext.items_present`；畫面綠框＋「盒內物品：…」。
- 檔案：`item_recognizer.py`（`ItemSeeds`、`build_trace`、`replay`、`build_dataset`、`train_command`、`ItemDetector`）、`tools/train_items.py`、
  `tests/test_items.py`；實驗腳本 `tools/dev/item_core.py`（延伸引擎）、`item_dataset_build.py`、`desk_item_seeds.py`、`desk_item_e2e.py`。
- 限制：後放的東西會蓋住先放的（鞋子蓋住標籤），所以只能判「這一輪看過」；物品換樣式要重框重訓；訓練中電腦很忙、記憶體要 3 GB 以上。
- 物件版實測（使用者桌面影片 178 秒、5 輪；我代替使用者框了 13 次 → 延伸成 122 張訓練／23 張驗證，第 5 輪完全不進訓練）：
  訓練 30 epoch 共 2074 秒；整支影片監控 **5 輪全部完成、每輪三樣物品都看到**（第 5 輪：標籤 15 格、鞋子 4 格、黑色小物 6 格），
  唯一一次「漏做：標籤」是錄影開頭桌上原本就有的那個盒子。ultralytics 回報的 val mAP50 只有 0.22——它算的是框的精確度，
  清單只需要「看到過」，所以兩個數字會差很多；監控時每格偵測 136～158 ms。
- 物件版取代狀態版：`apply_item_steps` 會把 `box_state` 步驟拿掉（狀態範例留著不影響）；兩版都在同一張盒內裁圖上跑。
- 使用者自己在「桌面測試」專案框了 25 次（黑色包裝袋 9／標籤 9／鞋子 7）訓出的第一版模型：鞋子 158 格只認到 1 格、標籤信心 p90 0.48。
  原因是延伸時樣板比對沒有位置限制，鞋子（大塊深色）配到陰影和手，標出來的框漂到別處（`cache/desk_overview/user_ds.jpg`）。
  修法：比對到的位置要跟框選位置重疊 IoU ≥ 0.3（`MATCH_MIN_IOU`，同一次擺放東西不會動），同名重疊框只留一個，空盒參考門檻放寬到 0.05；
  偵測門檻 0.5→0.3；訓練／驗證改隨機 15%（原本「最後一次擺放當驗證」把 150 張切掉 51 張）；產生物品步驟時清掉舊手勢步驟與 gesture_model。
  修完重訓（124 張、30 epoch、1422 秒）：val mAP50 0.99；整支影片 158 格抽樣中黑色包裝袋／標籤／鞋子各在 42／44／38 格 ≥ 0.5，
  信心中位數 0.90／0.96／0.79（鞋子原本 0.08）；監控 2 輪三樣全打勾、0 誤報（影片在第 2 輪拿走盒子前結束）。
- 第二版模型（2026-10-27 下午）：延伸與偵測都**不再等手離開**（使用者問「手一直在盒內、放完就蓋上」——原規則會整輪沒機會看），
  資料 134 張含手在盒內的格；重跑整支影片一樣 5 輪全完成，第 5 輪三樣各看到 20／19／9 格（舊模型 15／4／6）。
- 介面版警告要人按「重置／解除警告」，沒按整個狀態機卡住（使用者實測：開頭的「漏做：標籤」之後全部不算）→ 示範影片 3 秒、相機 15 秒自動解除並記「重置」。
- 「蓋著的盒子也被當成開著的內盒」（使用者實測）：蓋上的盒子 YOLO-World 一樣認成 cardboard box，區域還在、輪次不結束。
  解法：第 1 步多一顆「這一格是蓋著的盒子」（標 2～3 格），`OpenClosedClassifier` 用 CLIP 最近鄰比「空盒參考格（開著）」vs「蓋著格」，
  `BoxStateRunner.box_open`（最近 3 次投票一致才換），`FrameContext.box_open`；`object_absent` 條件＝盒子不在 **或** 蓋著；
  蓋著時不跑物品偵測。「開著」的範例要同時含空盒參考格和框選物品的那些格，否則裝滿黑鞋的盒子會被判成蓋著（第 2 輪 1 秒就假結算）。
  另外建立訓練資料時從空盒參考格推 `work_area`（聯集外擴 20%；40% 還會包到架上紙箱），每次產生步驟都重套——
  桌面測試影片右上角架上的紙箱（conf 0.21、中心 x 1537）會讓「盒子不在」永遠不成立，第 2 輪結不了。
  修完整支影片（介面同款背景偵測迴圈）：3 輪全結算——22.3 s（蓋上）、48.8 s（拿走）、67.1 s（蓋上後放著不動），每輪三樣都打勾。
- 「盒子拿走沒有結算」（使用者實測，介面有、命令列沒有）：介面用 AsyncDetector 每 0.2 秒偵測一次、命令列每 15 格一次，
  新盒子在舊盒子拿走後約 4 秒就放上來，介面較早偵測到它，把 2 秒的結束倒數打斷（`tools/dev/async_vs_sync_probe.py` 逐格對照），
  整輪永遠不結算。錨點本身已沿用 3 秒才算消失，`object_absent` 的 hold 改 0.5 秒（總延遲約 3.5 秒）；畫面加倒數文字。
- 相機延遲（使用者實測「串流跟現實延遲有點高」）：處理約 10～17 fps 追不上相機 30 fps，`cap.read()` 不丟格就越積越多。
  `CameraSource` 改成背景執行緒一直收、只留最新一格（實測慢消費者 80 ms/格：8 秒丟 141 格、每次 read 等 1 ms＝永遠拿最新），
  FFmpeg 加 `fflags;nobuffer|flags;low_delay|max_delay;500000`；錄影用 `latest_only=False` 保留每一格。
  物件版專案不再白跑 CLIP 狀態辨識（沒有 box_state 步驟就不載範例）。殘餘延遲＝相機編碼＋網路約 0.3～0.5 秒＋一次處理。

### 看整張桌面（固定範圍，2026-10-28）：物品不在盒子裡也能認

現場影片（俯拍 1080×1920、206 秒、8 輪）的流程跟桌面測試不一樣：
翻鞋舌檢查在上方大紙箱裡做（盒子還沒放上來）、條碼貼紙貼在盒端、盒子翻面看盒底小貼紙、**鞋子放在蓋著的盒子上面**再整組推走。
「跟著盒子裁盒內」的物件版認不到這些：盒內沒東西、鞋子放上去後 YOLO-World 找不到被遮住的盒子（實測 0～0.3）、檢查發生在盒子出現之前。
所以物件版多一種看法（使用者裁示「那就改成看整張桌面」）：

- 標記：第 1 步勾「看整張桌面（固定範圍，不跟著盒子）」→ 在畫面上框住整個工作桌面 →「把框選設為桌面範圍」（`seeds.region`，橘框）。
  「這一格是空盒」變成「這一格是空桌面」；框選的物品可以是任何動作後的樣子（檢查鞋舌＝手壓在紙箱裡的鞋上、藍盒、盒端貼紙、盒底朝上、鞋子在盒上）。
  盒子的各種樣子（立著、翻面、鞋子放在上面、貼了標籤）在右邊清單選它按「＝盒子的樣子」（`seeds.box_labels`）：看到任一個就當盒子還在桌上。
- 延伸：`_labels_fixed`——每個種子只往前後 2 秒（`FIXED_WINDOW_SEC`）用樣板比對（位置要跟框選時重疊），參考格＝最近的空桌面格；
  空桌面格本身當背景圖（沒有 txt）寫進資料集。整支影片比對既慢（桌面 1000×900 × 上千格）又會配到別輪同樣的東西。
  訓練與偵測都用 imgsz 960（`FIXED_IMGSZ`；桌面裁圖約 1000 px 寬，640 會把 100 px 的貼紙縮到 60 px）；`ItemDetector` 從模型檔的 `train_args` 讀訓練尺寸。
- 監控：`project.state_fixed`＝桌面範圍（`sync_item_region` 從 seeds 套過去；建資料與產生步驟前都套）。`BoxStateRunner.fixed_items`：
  每 N 格在固定裁圖上偵測，不看錨點新不新鮮、不等手離開；`box_seen`＝看到 box_labels 裡的物品。
  `object_absent`＝YOLO 找不到盒子 **且** 模型也沒看到盒子的樣子。清單模式加 `end_seen`（只在固定範圍模式）：這一輪看到過盒子之後
  「不在」才算推走——翻鞋舌檢查時盒子本來就不在，不能當結束、也不能擋住下一輪開始。`apply_item_steps` 在固定範圍時把盒子錨點的
  `max_age_sec` 縮到 1 秒、`cycle_arm_clear_sec` 0.2 秒（現場兩輪只隔 2.7～6 秒，沿用 3 秒會把下一雙的檢查算到上一輪）。
- 現場專案 `sop_projects/現場包裝1`：錨點 `blue box`（`cardboard box` 141/145 格都中上方大紙箱）、桌面範圍 (40,120,1040,900)、
  空桌面 6 格、種子 44 個（第 1～3 輪，座標從格線圖讀出來再用 `cache/site/seeds_check.jpg` 核對——第一次有 13 個框偏掉）；
  延伸成 128 張（122 張有標籤、6 張背景）：檢查鞋舌 20／藍盒 80／盒端貼紙 25／盒底朝上 12／鞋子在盒上 14。
- 現場影片實測（訓練 30 epoch／imgsz 960 共 3134 秒，約 100 秒一個 epoch；整支影片 6192 格跑 346 秒，不跑手部）：
  影片實際 7 輪完整＋1 輪未完（第 7 輪 190.8～196.8 s 真的沒貼標籤、沒翻面）。第 1～5 輪五項全打勾、5 輪全結算，
  第 4～5 輪完全沒進訓練；第 7 輪正確報「漏做：盒端貼紙、盒底朝上」；第 6 輪誤報「漏做：盒端貼紙」——標籤 184.0～184.6 s 看得到，
  模型連 conf 0.1 都沒框到（只有 5 個標籤種子、都是同一個角度）。第 4～7 輪分項：檢查鞋舌 4/4、藍盒 4/4、鞋子在盒上 4/4、
  盒底朝上 3/3、盒端貼紙 2/3。空檔 84 s 桌面被誤認成藍盒 0.7 秒 → `cycle_discarded`，沒報警。
  **已知弱點**：盒子側躺露出牛皮色上蓋時（180～183 s、202～205 s）會被當成「盒底朝上」，第 6 輪剛好後面真的翻了才沒出錯。
  **抓到一個邏輯 bug**：`end_seen` 原本輪與輪之間也會被空檔的盒子閃一下設成 True（84 s 誤判、157 s YOLO 閃一下），
  第 6 輪的翻鞋舌檢查被當成「結束條件成立中」沒開始新的一輪、整輪報漏做兩項；改成只在輪次進行中計、沒有輪次時結束條件一律不成立。
  「檢查鞋舌」這一類是外觀（手壓在鞋舌那帶、淺色內裡露出），**換鞋款不通用**，使用者指出鞋舌內側不一定淺色；
  後續方向是「手停在鞋子上 ≥ 1.2 s」（通用鞋子偵測＋手部關鍵點，區域跟著鞋子走、不怕鏡頭偏），必要時再接手勢模式判翻鞋舌動作。
  驗證腳本 `site_eval.py`（不跑 MediaPipe，物品步驟用不到手）、`site_probe_frames.py` 在 session scratchpad；時間軸圖 `cache/site/v1b_timeline.png`。

### 手停在物件上（hand_on_object，2026-10-28）：做完沒痕跡的動作，看手不看外觀

使用者裁示「檢查鞋舌改成手停在鞋子上的規則」（鞋舌內側不一定淺色，外觀訓練換鞋款不通用；區域要跟著鞋子、不綁紙箱、不怕鏡頭偏）。

- 步驟型別 `hand_on_object`：`object`＝錨點名稱，條件＝某隻手 4 個關鍵點裡 ≥ 2 個落在該錨點任一框內（`hands_in_inner` shrink 0）；
  畫面上的區域就是物件現在的框。物件用 YOLO-World 通用提示找（`shoe`），不用框、不用訓練。
- 步驟欄位 `continuous`（清單模式）：dwell 要**連續**停留（中斷 ≤ `CONTINUOUS_GAP_SEC` 0.3 秒算同一段），不是整輪累加——
  手經過鞋子的零碎片段（靜置時約 5% 取樣）整輪加起來也會超過門檻。`streak_sec`／`gap_sec` 每輪歸零；畫面進度條顯示目前這一段。
- 標記頁第 1 步多一列「手停在物件上的步驟：名稱／物件（英文）／連續停留秒數 → 加入」（`seeds.hand_steps`）；`apply_item_steps`
  對同名物品不再產生 item 步驟，並自動補錨點（名稱＝提示，min_conf 0.15、max_count 2、min_area 8000、max_age 2.5 秒）。
- 現場影片量測（`site_shoe_probe.py`，YOLO-World 每 5 格、手從 trace.json）：提示 `shoe` 在檢查片段 ≥1 鞋 61%、≥2 鞋 49%、conf p50 0.29，
  `sneaker`／`footwear` 都不行（21%／4%）；手在鞋框內：檢查片段 58%，沿用 2.5 秒後 **97%**；搬鞋 83%（但只持續 0.3～0.6 秒）；
  靜置 5%；空紙箱 0%。門檻用連續 0.7 秒：檢查 0.8～2.5 秒 > 0.7 > 搬鞋 0.6 秒，邊界只有 0.1～0.2 秒，換人操作要重看。
- 實測（v3，YOLO 每 10 格、兩個提示；6192 格 397 秒）：**檢查鞋舌 8 輪全對**（含影片結尾第 8 輪），其餘四項與 v1b 相同
  （第 1～5 輪全結算、第 7 輪正確報漏做兩項、第 6 輪仍誤報盒端貼紙）。第一版 min_conf 0.25 時第 2 輪整段 shoe 只有 0.10～0.23，
  沒錨點 → 報漏做；降到 0.15 才過。
- `tools/train_items.py --device auto`：有 CUDA 就用 GPU。資料集在 `cache/item_ds_<雜湊>/`（純 ASCII），把那個資料夾＋
  `tools/train_items.py`＋`models/yolov8n.pt` 搬到有 GPU 的電腦（同版 ultralytics 8.3.x），跑完把 `best.pt` 拷回專案 `items/model/best.pt`；
  128 張 30 epoch imgsz 960 在這台 CPU 要 3134 秒，一般 GPU 約 2～5 分鐘（未實測）。

### 第 6 輪誤報「盒端貼紙」：原因與重新訓練步驟（2026-10-29）

舊 CPU 模型跑示範影片 15 段，第 6 輪有 7 次報「缺盒端貼紙」。對照影片（第 6 輪結算在第 5709 格、190.2 秒，作業在 176～190 秒）：

1. **貼紙落在桌面範圍外**：`state_fixed` = `[40, 120, 1040, 900]`，下緣在 y = 1020。第 6 輪盒子放得低，184～186 秒貼盒端貼紙時，
   貼紙那一角壓在下緣或超出，裁圖時被切掉，模型看不到。
2. **角度沒看過**：`seeds.json` 的標記全在第 850～2008 格（28～67 秒，影片共 206 秒）；盒端貼紙 5 個手標都在 33／48／62 秒，
   盒子都斜放。第 6 輪盒子橫放，模型沒看過。`train_items.py` 原本沒開旋轉（ultralytics 預設 `degrees=0`、`flipud=0`）。

GPU 電腦上的步驟：

1. `git pull` 取得新版 `tools/train_items.py`（預設 `--degrees 45 --flipud 0.5`）。
2. 「標記物品」頁：把**桌面範圍**往下拉大，讓盒子放低時整個盒子仍在範圍內（例如下緣拉到 y ≈ 1220）。範圍改了裁圖就不同，一定要重訓。
3. 「標記物品」頁：補標第 6 輪的盒端貼紙（約第 5520～5580 格），有其他橫放的盒子也一起補。補標比旋轉增強可靠。
4. 重新訓練。介面的「訓練」會自動帶新參數；手動跑時：
   `python tools/train_items.py <cache/item_ds_…/data.yaml> <輸出資料夾> --imgsz 960 --epochs 60 --copy-to sop_projects/<專案>/items/model/best.pt`
   （增強變多，30 epoch 可能不夠，先用 60；旋轉太大框會變鬆，小貼紙認不準時把 `--degrees` 降到 30）。
5. 用示範影片跑網頁版，確認第 6 輪不再報缺盒端貼紙、第 1～5 輪沒有變差。

## 網頁版（2026-10-28，第一階段：監控頁）：`start_web.bat` → 瀏覽器開 http://<這台的IP>:8765/

使用者裁示「先做監控頁，好了後再做標記頁，要讓使用者簡單操作」。辨識核心完全不動，只是把介面③換成伺服器＋網頁：

- `monitor_service.py`：`MonitorService`，`main_ui.MonitorWorker` 的無 Qt 版（同樣的載入階段訊息、示範影片照原速、警告自動解除
  示範 3 秒／相機 15 秒）。畫面疊字後編碼一次 JPEG，所有瀏覽器共用；`snapshot()` 給清單／輪數／警告／最近 60 筆事件。
  示範影片處理跟不上時跳格（每處理 1 格最多跳 2 格），現場 1080×1920 影片這台只有 5～7 fps，不跳格 206 秒要播 20 分鐘。
- `web_server.py`：純標準函式庫 `ThreadingHTTPServer`（venv 沒裝 FastAPI，且不想再多裝）。路由：`/`、`/stream.mjpg`（MJPEG）、
  `/api/state`、`/api/projects`、`POST /api/start|stop|reset|screenshot`。同資料夾放 `web_password.txt` 就要求密碼（HTTP Basic，區網用）。
  相機來源沿用 `camera_source.txt`（只留在伺服器這台）。
- `web/index.html`：一頁、不用框架。左邊大畫面，右邊：專案下拉（沒有步驟的灰掉）、示範影片／相機、一顆「啟動／停止監控」、
  完成／漏做輪數、步驟清單（✓ 與停留進度）、紅色警告框＋大顆「解除警告」、事件列表、存截圖。手機寬度會改成直排。
  每 0.5 秒輪詢 `/api/state`；新警告時用 Web Audio 嗶一聲（瀏覽器規定要先點過頁面一次才會有聲音）。
- 測試 `tests/test_web.py`（替身服務：路由、串流第一張、密碼）；實機 smoke（現場專案示範影片）：載入 16 秒後出畫面、JPEG 約 100 KB、
  啟動／重置／截圖／停止都正常。
- 限制：一台伺服器一次一個專案一支相機（CPU 1～2 核／支）；訓練時仍不能同時監控。
- 串流 FPS＝辨識處理速度（每格跑完整套才送出）。現場專案示範影片每格實測（`profile_loop.py`）：手部 57 ms／物品模型（每 5 格、960）
  平均 43 ms／疊字 19 ms／JPEG 3 ms，共 123 ms → 8.1 fps，YOLO-World 背景 185 ms 一次還會搶 CPU。
  最快的改法：手部改成**所有來源**每 2 格一次（原本只有相機如此，影片每格都做），且 `Project.needs_hands` 為 False 的專案
  （純物件版、固定桌面範圍、沒有手部步驟）整個跳過 MediaPipe → 同一段 98 ms → 10.2 fps。
  第二步「顯示與辨識分開」（2026-10-28 做了）：`MonitorService._display_loop` 另一條執行緒用 `CameraSource.latest()`（不等待、不算拿走）
  拿相機每一格，疊上辨識執行緒最近一次的結果（`_live`＝ctx／fsm／runner／辨識 fps）後編碼送出，上限 `DISPLAY_FPS` 25；
  辨識執行緒只更新狀態（`_publish_state`），不再畫圖。只有相機來源這樣做，影片檔仍是處理完的格直接送。
  配套：`draw_texts` 改成每一行字第一次用 PIL 畫成小圖快取、之後 numpy 貼圖（整張 PIL 來回轉換才是貴的），疊字從 19 ms 降到幾 ms。
  實測桌上 1920×1080 相機＋桌面測試專案：瀏覽器端收到 23.7 fps（上限 25），辨識 26.9 fps；畫面左上角顯示「辨識 x fps  畫面 y fps」。
- **取樣密度要跟來源時間走，不能跟處理格數走**（2026-10-28 下午）：現場專案走網頁示範影片（照原速、跟不上就跳格）第一次重驗
  只處理 2067/6192 格，物品模型「每 5 個處理格」變成每 0.5 秒一次、手部每 0.2 秒一次 → 第 2 輪漏盒底朝上（只露 1 秒）、第 3 輪漏檢查鞋舌、
  第 7 輪開始得太晚被當「只持續 3 秒」丟掉。改法：`BoxStateRunner.update(..., t_now=)` 以 `every/fps` 秒為間隔（`_due`），
  停好判定也改用時間；手部偵測改「處理快於 16 fps 才隔格」（`HAND_MIN_INTERVAL_SEC` 0.06）。
  改完**離線逐格跑（site_eval，條件同 v3）結果與 v3 完全相同**（5 輪結算、第 6 輪誤報盒端貼紙、第 7 輪正確報漏做兩項）；
  但網頁照原速跑在這台只有 4.7 fps、追不上 30 fps 影片（400 秒才播到 181 秒），物品四項全對、檢查鞋舌第 2、3 輪仍漏——
  這是 CPU 不夠（手部 60 ms＋物品模型 130 ms＋YOLO-World 背景 300 ms 一次搶 CPU），不是演算法。現場 1080×1920 直式畫面要即時跑，
  要更強的 CPU 或 GPU，或把相機降到 720×1280 重驗。
- YOLO-World 節流（三個監控迴圈都改成每 `detect_every_n_frames÷fps` 秒才送一次，現場專案設 15 格＝0.5 秒）：離線逐格結果不變
  （檢查鞋舌 8/8），但網頁原速只從 4.7 → 5.0 fps，檢查鞋舌第 2、3 輪仍漏。真正的瓶頸是物品模型：處理慢時 `_due` 讓它每格都跑，
  imgsz 960 一次約 200 ms。YOLO-World 不是主因（我原本估 7～8 fps 估錯了）。
- **物品模型改用 OpenVINO**（`item_recognizer.openvino_model`，`ItemDetector.backend`）：載入時自動把 best.pt 轉成 OpenVINO
  （約 11 秒，快取在 `cache/ov_<雜湊>/`，重訓後自動重轉；沒裝 openvino 或失敗就照用 PyTorch）。i5-1235U、960 實測 446 → 61 ms，
  40 張抽樣認出的東西完全相同。venv 裝 `openvino==2024.6.0`（2026 版拿掉了 `openvino.runtime`，ultralytics 8.3.40 匯出會炸），
  連帶 numpy 2.2→2.1.3（pip 警告 jax 要 ≥2.2，本專案沒用 jax，70 個測試照過）。
  之後每格（慢速路徑：手部與物品模型每格都跑、影片跳格）：讀檔解碼 36／手部 84／物品 44／疊字 11／JPEG 4 ms → 7.0 fps。
  現在最貴的是 MediaPipe 手部。相機來源時解碼在收影像的執行緒，辨識估約 9 fps；網頁原速跑影片檔 6.1 fps（檔案解碼＋跳格在同一條）。
- 試過但不採用：手部輸入縮到 480／384（28～32 ms 都一樣，MediaPipe 內部本來就縮小）、手部只看桌面範圍 +5%（一樣 28 ms，抓到手 312→295/315）、
  YOLO-World 轉 OpenVINO（固定 640×640 方圖，118 → 381 ms 反而變慢）。手部是 MediaPipe 自有格式，不能轉 OpenVINO。
- 採用：監控時 `limit_torch_threads()`（`MONITOR_TORCH_THREADS` 1）。三個模型單獨跑都不慢，慢在互搶 CPU；torch 1／2／4／6 執行緒
  實測 10.5～11.3／9.9～10.3／9.8／9.0 fps。網頁原速跑現場影片 6.1 → 7.6 fps，判斷結果和上一次相同（第 2、3 輪仍漏檢查鞋舌）。

### 框選後自動往前後追（2026-10-28）

使用者要求「框選後直接抓前後幾秒自動框選再訓練」。原本延伸藏在建資料集裡、看不到也刪不掉，改成框完當下就追：
- `item_recognizer.track_seed`：前後各 `window_sec`（預設 2 秒、網頁可設 0～5），每 3 格在上一個位置外擴 50% 範圍做樣板比對，
  比原框 ≥ 0.55 或比上一個位置 ≥ 0.75 才收，連續 2 次不夠就停；影片那段只循序讀一次。現場影片一次 2～3 秒。
- `Seed.auto`／`Seed.group`：自動框和手框同一組；`remove_seed` 刪手框會整組刪、刪自動框只刪那一格；`counts()` 只算手框，`auto_counts()` 另計。
- 建資料集：`split_seeds` → 追過的那一組直接當標籤、不再延伸（不然使用者刪掉的錯框又會長回來）；舊資料（沒追過）照舊延伸。
- 網頁標記頁：虛線淡綠框＝自動框、「◀ 上一個框／下一個框 ▶」逐格檢查、影格下方 ✕ 刪單格、「刪除這次的自動框」整批刪。
  桌面程式也會自動追 2 秒（淡綠框），但刪單格要用網頁。
- 現場實測（`track_real.py`，4 個原有手框種子）：盒端貼紙 15 格、鞋子在盒上 7 格、盒底朝上 5 格都框得準；藍盒 28 格裡往前 0.8 秒那幾格
  （盒子還在手上移動）框偏到手上——這正是要讓使用者看過再刪的原因。
- 現場專案 34 個手框全部改新方式：追出 559 格、人工逐組看過刪 92 格（追進翻面／還在手上的階段）→ 467 格（備份 seeds_before_retrack.json，未重訓）。
- 追蹤加顏色檢查（`TRACK_COLOR_MIN` 0.5，HSV 直方圖跟手框那一格比）：拿上面人工判過的 559 格當答案，
  錯框 92 → 13、正確框 467 → 418；只限制移動距離幾乎沒用。代價：每次加框追蹤從約 2.6 秒變約 4.8 秒。

### 第二階段：標記頁 http://<IP>:8765/label（2026-10-28）

- `label_service.py`：`LabelService`（影格 JPEG、專案資訊、種子操作 `seed_action`、`generate_steps`、新增專案、上傳匯入）、
  `TrainJob`（背景：trace → 資料集 → 子程序 `tools/train_items.py`，進度從 stdout 的「epoch N/M」抓）、`RecordJob`（開相機、
  用相機解析度寫回 config、錄 demo.avi、預覽 JPEG）。資料格式與桌面程式相同，`items/seeds.json`／`config.json` 兩邊共用。
- 路由：`GET /label`、`/api/project?name=`、`/api/frame?project=&n=[&w=]`（整張原尺寸 JPEG，框選座標直接用畫面座標）、
  `POST /api/seeds`（action：add_ref／remove_ref／add_closed／remove_closed／add_seed／remove_seed／remove_item／set_region／
  set_box_label／add_hand_step／remove_hand_step）、`POST /api/project/create`、`POST /api/project/upload?name=`（multipart，
  自己解析，不用 cgi）、`POST /api/train/start`＋`GET /api/train/state`、`POST /api/steps/generate`、
  `POST /api/record/start|begin|stop`＋`GET /api/record/state`＋`/api/record/stream.mjpg`。
- `web/label.html`：沒有影片的專案先出現「A. 用相機錄一段／B. 上傳影片」；有影片就是左邊影格＋canvas 拖拉框選（滑鼠或手指）、
  ◀◀◀ ◀◀ ◀ 滑桿 ▶ ▶▶ ▶▶▶、鍵盤 ← →（Shift 一次 30 格），右邊和桌面程式一樣的三步（標記→訓練→產生步驟）與同一套提示文字。
  步驟按鈕都是「切換」：同一格再按一次「空盒」就取消。這一格的框選列在影格下方，可逐個 ✕。
- 測試 `tests/test_web.py` 7 個（含合成影片的種子操作、建專案＋上傳、multipart 解析）。


## 手勢特徵版本 2（2026-10-22）

使用者回報「拿東西移動」的動作認不準（約 60%）。原因：版本 1 只有手形＋手相對錨點的位置，沒設錨點時模型看不到手在畫面裡怎麼移動；另外手握著東西時 MediaPipe 常丟手，視窗「有手比例」門檻 0.5 太嚴，很多視窗直接沒機率。版本 2 改：

- 新增手腕在畫面中的絕對位置、視窗內的平均／最大速度、淨位移、路徑長、有手比例、雙手距離（`gesture_classifier.py`，`FEATURE_VERSION = 2`，`FRAME_DIM` 99）。
- 「有手比例」門檻 `MIN_HAND_RATIO` 0.5 → 0.3。
- 模型檔記錄特徵版本，舊模型載入會提示「請重新訓練」。

同一支鞋盒影片、同一份 24 段標籤的 5 組交叉驗證：**78% → 82%**；拿鞋盒 0.68 → 0.79、放防黴片 0.94 → 0.97、放鞋子 0.92 → 0.95、開鞋盒 0.67 → 0.64（開盒時手被遮住，片段內只有 52% 的格數抓得到手）。外盒片段資料 83% → 81%（持平）。再往上的三個方向：把鞋盒設成錨點並加進 `gesture_anchors`（手相對鞋盒的位置）、MediaPipe 的 `min_hand_detection_confidence` 0.6 → 0.4、多錄幾輪。

## 手勢訓練（第 2 批工具）：每個動作錄一段影片，檔名就是標籤

```powershell
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8   # 不設的話中文輸出會變亂碼
# a. 錄片段：空白鍵開始／停止一段，n 下一個動作，q 離開。存到 sop_projects\<專案>\gestures\<動作名>\<動作名>_001.avi
& $PY tools\record_clips.py --project sop_projects\桌面測試 --labels "確認尺碼,放鞋墊,其他" --camera 0

# b. 訓練：抽特徵（快取在 gestures\_cache，重訓不用重抽）→ 依片段分組交叉驗證 → 存 gesture_model.pkl 並寫進 config
& $PY tools\train_gesture_clips.py --project sop_projects\桌面測試 --yolo $W
```

- 每個動作建議 15～30 段、每段 2～5 秒；少於 5 段的動作會被跳過（`--min-clips`）。
- 一定要錄一類「其他」：空手移動、整理桌面等不是目標動作的手部動作，分類器才知道「不是」長什麼樣。
- 訓練完在 config 的 `steps` 加 `{"type": "gesture", "label": "確認尺碼", "min_prob": 0.5, "dwell_time_sec": 0.5, ...}`。
- 驗證方式是「整段影片一起當測試」，同一段的前後視窗不會互相洩題，數字比逐格標籤的版本可信。
- 用外盒影片切出的 19 段假片段跑過一次：5 組交叉驗證 83%（尺码确认 召回 0.86、鞋垫 0.78、其他 0.88）；「标签确认」只有 3 段被自動排除。

## 工作流程（第一版：整支影片＋逐格標籤）

```powershell
cd <專案資料夾>
$env:PYTHONUTF8='1'
$PY=".venv\Scripts\python.exe"
$W="models\yolov8s-worldv2.pt"

# 1. 抽特徵成快取（一支 1080p 影片約 15 分鐘；手部逐格、YOLO 每 15 格）
& $PY tools\extract_features.py sop_projects\範例_外盒\demo.mp4 cache\outer.npz --yolo $W --prompts "cardboard box,sneaker" --every 15

# 2. 訓練手勢分類器（需要逐格標籤 csv：frame,label）
& $PY tools\train_gesture.py cache\outer.npz cache\outer_labels.csv sop_projects\範例_外盒_v3\gesture_model.pkl --anchors "sneaker,cardboard box"

# 3. 離線跑狀態機（讀快取，秒級）
& $PY tools\run_cached.py cache\outer.npz sop_projects\範例_外盒_v3 [--print-all]

# 4. 線上／整支影片即時跑（含偵測，慢）
& $PY core_detector.py --project sop_projects\範例_外盒_v3 --source sop_projects\範例_外盒\demo.mp4 --yolo $W --show

# 測試
.venv\Scripts\python.exe -m pytest tests -q
```

`cache\outer_labels.csv` 是從阿丘 demo 畫面上的「正在执行动作: 'X'」文字自動歸組來的（腳本在 session 暫存區，沒進專案）；換成自己的影片要另外標。

## config.json（schema 2）重點欄位

```json
"anchors": [{"name":"box","prompt":"cardboard box","pick":"largest","min_area":40000,"exclude_containing":["shoes"]},
            {"name":"shoes","prompt":"sneaker","pick":"all","max_count":2}],
"zones":   [{"name":"shoes_zone","anchor":"shoes"}, {"name":"box_zone","anchor":"box"},
            {"name":"label_zone","anchor":"box","rel":[0.5,0,0.5,0.5]}],
"cycle_end": {"type":"object_absent","object":"shoes","hold_sec":2.0},
"cycle_timeout_sec": 90.0,
"cycle_min_sec": 10.0,
"gesture_model": "gesture_model.pkl", "gesture_anchors": ["shoes","box"],
"steps": [
  {"step_index":0,"step_name":"處理鞋子","type":"hand_dwell","zone":"shoes_zone","dwell_time_sec":0.4,"timeout_sec":30},
  {"step_index":1,"step_name":"確認尺碼","type":"gesture","label":"尺码确认","min_prob":0.5,"dwell_time_sec":0.5,"timeout_sec":30},
  {"step_index":2,"step_name":"確認鞋盒標籤","type":"hand_dwell","zone":"box_zone","dwell_time_sec":0.4,"timeout_sec":30,"required":true}
]
```

- `exclude_containing`：候選框若包住另一個錨點的框（面積比 ≥ 0.7）就不算。用來排除「鞋子底下那片紙板」被當成鞋盒。
- `work_area`：偵測框中心不在裡面就丟掉（排除操作者腳上的鞋）。
- 有 `cycle_end` 就是清單模式；沒有就是依序模式（`skip_alert_sec`、每步 `timeout_sec` 才有作用）。
- `required: false` 的步驟沒做不報。

## 疑難排解

- **`OSError: [WinError 1114] ... torch\lib\c10.dll`**：PyTorch 的 DLL 在 Qt（PySide6）之後載入就會失敗。`main_ui.py` 與 `tools\e2e_walkthrough.py` 已固定「先 torch 後 Qt」；自己寫的腳本若同時用到 Qt 與 YOLO，`import torch` 要放在任何 `PySide6` import 之前。
- **`start_ui.bat` 點兩下沒反應或出現 `'cho' 不是內部或外部命令`**：批次檔裡不能有中文（cmd 切編碼後會錯位解析），本專案的 .bat 已改純英文。
- **介面第一次開要 10～20 秒**：在載入 PyTorch 與 MediaPipe，黑色命令視窗不要關。
- **PowerShell 抓工具輸出是亂碼**：先執行 `[Console]::OutputEncoding=[System.Text.Encoding]::UTF8`。

## 已知限制

- 標籤／QC 貼紙的「內容」驗不了（1080p 在這個距離條碼解不出來）；系統只能驗「手有沒有到標籤區」。
- 零訓練偵測每格約 0.2～0.4 秒（i5-1235U；單獨跑 194 ms、與別的工作同時跑 400 ms），所以 `detect_every_n_frames` 預設 15。單一 process 內逐格 MediaPipe（40 ms）＋每 15 格 YOLO（194 ms）平均每格約 53 ms，約 19 FPS；兩支影片同時抽特徵會互搶 CPU 慢 3 倍，一次跑一支。
- 手勢分類需要每個動作至少幾十個樣本；本次「左鞋」只有 1.5 秒的資料，不可用。
- `demo.mp4` 的旋轉標記是 270°，`video_recorder.open_video_oriented` 會處理；自己錄的影片要確認方向。

## 檔案

| 檔案 | 內容 |
| --- | --- |
| `project_store.py` | config 讀寫驗證（schema 1／2） |
| `scene.py` | 錨點、區域、重疊計算 |
| `object_detector.py` | YOLO-World 包裝、`AsyncDetector` 背景執行緒（只留最新一張，忙不過來就丟） |
| `gesture_classifier.py` | 手勢特徵、分類器、即時串流 |
| `gesture_data.py` | 片段 → 視窗資料集、依片段分組交叉驗證、報表 |
| `core_detector.py` | HandTracker、SOPStateMachine（依序／清單）、開發入口 |
| `video_recorder.py` | `CameraSource`（實際解析度、實測 fps）、`Recorder`（MJPG avi）、`import_demo_video`、影片開檔轉正、影格讀取、含中文路徑的圖片讀寫 |
| `tools/` | `anchor_probe`（調 prompt）、`new_project`（從相機建專案）、`record_clips`（錄手勢片段）、`train_gesture_clips`（片段訓練）、`extract_features`、`train_gesture`（逐格標籤版）、`run_cached` |
| `main_ui.py` | PySide6 操作介面（錄製／分段說明／監控），背景工作用 QThread |
| `training.py` | 「示範影片＋分段 → 模型 → 步驟」服務層，介面與命令列共用 |
| `tests/` | 39 條（v1 14、v2 14、live 3、gesture_data 4、training 3、ui 1） |
| `sop_projects/` | `範例_外盒`（schema 1）、`_v2a`／`_v2`（依序）、`_v3a`／`_v3`（清單）、`範例_內盒`（schema 1）、`_v2`（依序）、`_v3`／`_v3b`（清單） |
| `cache/` | `outer.npz`、`inner.npz`、`outer_labels.csv`、各次執行 log |
