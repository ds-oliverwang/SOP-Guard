# 模型檔

本資料夾要放 MediaPipe 手部關鍵點模型 `hand_landmarker.task`（約 7.8 MB），程式啟動時會讀：

```
models/hand_landmarker.task
```

下載網址：

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

PowerShell 下載指令（在專案根目錄執行）：

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "models\hand_landmarker.task"
```

找不到這個檔案時，程式會顯示完整路徑的錯誤訊息，不會閃退。

專案有設錨點（anchors）時還要 YOLO-World 權重 `models/yolov8s-worldv2.pt`（約 25 MB），Ultralytics 官方發布：

```powershell
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt" -OutFile "models\yolov8s-worldv2.pt"
```

權重檔都不放進 git（見 `.gitignore`）。
