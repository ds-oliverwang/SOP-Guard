"""訓練物品偵測小模型（yolov8n 微調；有 CUDA 用 GPU，否則 CPU）。
用法：python tools/train_items.py <data.yaml> <輸出資料夾> [--epochs 40] [--imgsz 640] [--batch 8] [--degrees 45] [--flipud 0.5]
輸出：<輸出資料夾>/best.pt 與 train.log；進度每個 epoch 印一行（介面用來顯示）。

角度增強（--degrees／--flipud）：標記通常只來自示範影片前幾輪，盒子擺的角度都一樣；後面某一輪盒子橫放就認不出來
（示範影片第 6 輪的盒端貼紙）。訓練時把圖隨機轉 ±degrees 度、上下翻轉，讓模型看過別的擺法。
注意：這只解決「角度沒看過」；物品落在桌面範圍（state_fixed）外面會被裁掉，要在「標記物品」頁把範圍拉大。"""
import argparse
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--base", default=os.path.join(ROOT, "models", "yolov8n.pt"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--copy-to", default=None, help="訓練完把 best.pt 複製到這個路徑（可含中文；訓練本身要走純 ASCII 路徑）")
    ap.add_argument("--device", default="auto", help="auto＝有 CUDA 就用 GPU，否則 CPU；也可指定 cpu／0。資料集資料夾＋這支腳本搬到有 GPU 的電腦跑完，把 best.pt 拷回專案 items/model/ 即可")
    ap.add_argument("--degrees", type=float, default=45.0,
                    help="訓練時隨機旋轉 ±N 度（ultralytics 預設 0＝不轉）。框是水平長方形，轉太大框會變鬆，小貼紙先用 45，不夠再加")
    ap.add_argument("--flipud", type=float, default=0.5, help="上下翻轉的機率（ultralytics 預設 0）；左右翻轉 fliplr 維持預設 0.5")
    a = ap.parse_args()
    import numpy as np
    if not hasattr(np, "trapz"):                 # numpy 2.x 拿掉 trapz，ultralytics 8.3.40 算 mAP 還在用
        np.trapz = np.trapezoid
    import torch  # noqa
    from ultralytics import YOLO
    device = a.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    use_cuda = device != "cpu" and torch.cuda.is_available()
    batch = a.batch
    if batch is None:
        if use_cuda:
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            batch = 16 if mem_gb >= 12 else 8
        else:
            batch = 8
    workers = a.workers
    if workers is None:
        workers = 0 if sys.platform == "win32" else min(4, max(1, (os.cpu_count() or 2) // 2))
    print(f"device: {device} (batch={batch}, workers={workers}, degrees=±{a.degrees:g}, flipud={a.flipud:g})", flush=True)
    if not use_cuda:
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    os.makedirs(a.out, exist_ok=True)
    run_dir = os.path.join(a.out, "run")
    shutil.rmtree(run_dir, ignore_errors=True)
    t0 = time.perf_counter()
    model = YOLO(a.base)
    model.add_callback("on_train_epoch_end", lambda tr: print(f"epoch {tr.epoch + 1}/{tr.epochs} 完成，已用 {time.perf_counter() - t0:.0f} 秒", flush=True))
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=batch, device=device, workers=workers,
                project=a.out, name="run", exist_ok=True, verbose=False, plots=False, val=True, patience=0,
                degrees=a.degrees, flipud=a.flipud)
    best = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.isfile(best):
        print("訓練結束但沒有產生 best.pt", flush=True); return 2
    shutil.copyfile(best, os.path.join(a.out, "best.pt"))
    if a.copy_to:
        os.makedirs(os.path.dirname(os.path.abspath(a.copy_to)), exist_ok=True)
        shutil.copyfile(best, a.copy_to)
    print(f"訓練完成：{a.epochs} 個 epoch，共 {time.perf_counter() - t0:.0f} 秒；模型 {a.copy_to or os.path.join(a.out, 'best.pt')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
