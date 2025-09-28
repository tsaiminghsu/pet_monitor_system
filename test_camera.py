# test_camera.py
import cv2
import time
import argparse

PREFERRED_RES = (1280, 720)
SCAN_RANGE = range(0, 6)  # 依需要擴大

BACKENDS = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("DEFAULT", 0),
]

def try_open(index: int):
    print(f"\n=== 測試 Camera index={index} ===")
    for name, backend in BACKENDS:
        cap = cv2.VideoCapture(index, backend) if backend != 0 else cv2.VideoCapture(index)
        ok = cap.isOpened()
        print(f"  - 後端 {name}: isOpened={ok}")
        if not ok:
            cap.release()
            continue

        # 嘗試設定解析度
        w, h = PREFERRED_RES
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        time.sleep(0.2)  # 給驅動一點時間

        # 讀取幾次避免第一張黑畫面
        ret, frame = False, None
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                break
            time.sleep(0.05)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"    實際解析度: {actual_w}x{actual_h}, FPS(報告值): {fps:.2f}")
        print(f"    讀取結果: ret={ret}, frame={'OK' if frame is not None else 'None'}")

        if ret and frame is not None:
            snap_name = f"snapshot_{index}_{name}.jpg"
            cv2.imwrite(snap_name, frame)
            print(f"    ✅ 已儲存快照 -> {snap_name}")
            cap.release()
            return True, name, (actual_w, actual_h)

        cap.release()

    print(f"    ❌ index={index} 無法取得有效影像")
    return False, None, (0, 0)

def list_all():
    print("開始掃描攝影機...")
    results = []
    for i in SCAN_RANGE:
        ok, backend, res = try_open(i)
        results.append((i, ok, backend, res))
    print("\n=== 掃描總結 ===")
    for i, ok, backend, (w, h) in results:
        if ok:
            print(f"  index={i}: ✅ 可用（{backend}），實際 {w}x{h}")
        else:
            print(f"  index={i}: ❌ 不可用")
    return results

def quick_test(index: int):
    ok, backend, res = try_open(index)
    if ok:
        print(f"\n✅ Camera {index} 測試通過，後端 {backend}，解析度 {res[0]}x{res[1]}")
        return 0
    else:
        print(f"\n❌ Camera {index} 測試失敗，請檢查是否被其他程式占用或更換 index/後端")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="測試攝影機是否可用，並存快照。")
    parser.add_argument("--index", type=int, default=None, help="指定相機 index（不指定則掃描）")
    args = parser.parse_args()

    if args.index is None:
        list_all()
    else:
        exit(quick_test(args.index))
