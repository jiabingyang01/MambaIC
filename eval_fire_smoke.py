import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(root_dir: Path):
    for path in sorted(root_dir.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTS:
            yield path


def resolve_model_path(args):
    if args.model:
        return args.model
    if not args.hf_repo:
        raise SystemExit("Provide --model or --hf-repo")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for --hf-repo. "
            "Install it or pass --model with a local .pt file."
        ) from exc
    return hf_hub_download(repo_id=args.hf_repo, filename=args.hf_filename)


def summarize_stats(stats):
    images = stats["images"]
    stats["detections_per_image"] = stats["detections"] / images if images else 0.0
    stats["images_with_det_ratio"] = stats["images_with_det"] / images if images else 0.0
    stats["per_class_conf_mean"] = {}
    for name, count in stats["per_class_counts"].items():
        total_conf = stats["per_class_conf_sum"][name]
        stats["per_class_conf_mean"][name] = total_conf / count if count else 0.0
    stats.pop("per_class_conf_sum", None)
    return stats


def run_on_dir(model, img_dir, args, run_name, out_dir):
    img_dir = Path(img_dir)
    if not img_dir.exists():
        return {"name": run_name, "error": f"missing dir: {img_dir}"}

    image_count = sum(1 for _ in iter_images(img_dir))
    if image_count == 0:
        return {"name": run_name, "error": f"no images found in: {img_dir}"}

    stats = {
        "name": run_name,
        "dir": str(img_dir),
        "images": 0,
        "images_with_det": 0,
        "detections": 0,
        "per_class_counts": defaultdict(int),
        "per_class_conf_sum": defaultdict(float),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": args.device,
    }

    predict_kwargs = {
        "source": str(img_dir),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": args.device,
        "stream": True,
        "verbose": False,
        "save": bool(args.save_vis),
    }
    if args.save_vis:
        predict_kwargs["project"] = str(out_dir)
        predict_kwargs["name"] = run_name

    start = time.time()
    for result in model.predict(**predict_kwargs):
        stats["images"] += 1
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        stats["images_with_det"] += 1
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        for c, score in zip(cls, conf):
            class_name = result.names[int(c)]
            stats["detections"] += 1
            stats["per_class_counts"][class_name] += 1
            stats["per_class_conf_sum"][class_name] += float(score)

    stats["seconds_total"] = time.time() - start
    stats["seconds_per_image"] = stats["seconds_total"] / stats["images"] if stats["images"] else 0.0
    return summarize_stats(stats)


def print_summary(stats):
    if "error" in stats:
        print(f"{stats['name']}: {stats['error']}")
        return
    print(
        f"{stats['name']}: images={stats['images']} "
        f"with_det={stats['images_with_det']} "
        f"ratio={stats['images_with_det_ratio']:.3f} "
        f"dets={stats['detections']} "
        f"dets/img={stats['detections_per_image']:.3f} "
        f"sec/img={stats['seconds_per_image']:.4f}"
    )
    if stats["per_class_counts"]:
        for name in sorted(stats["per_class_counts"]):
            count = stats["per_class_counts"][name]
            mean_conf = stats["per_class_conf_mean"].get(name, 0.0)
            print(f"  {name}: count={count} mean_conf={mean_conf:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fire/Smoke detection eval on multiple dirs.")
    parser.add_argument("--model", type=str, default="", help="Local .pt model path.")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="TommyNgx/YOLOv10-Fire-and-Smoke-Detection",
        help="HF repo ID to download best.pt if --model not set.",
    )
    parser.add_argument("--hf-filename", type=str, default="best.pt", help="HF filename.")
    parser.add_argument("--original", type=str, default="/home/zhaorun/zichen/yjb/projects/CV/MambaIC/dataset/all")
    parser.add_argument("--jpeg", type=str, default="/home/zhaorun/zichen/yjb/projects/CV/MambaIC/output/JPEG/10")
    parser.add_argument("--mambaic", type=str, default="/home/zhaorun/zichen/yjb/projects/CV/MambaIC/output/MambaIC/0.008/all")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="", help="cuda device or cpu.")
    parser.add_argument("--save-vis", action="store_true", help="Save annotated images.")
    parser.add_argument("--out-dir", type=str, default="/home/zhaorun/zichen/yjb/projects/CV/MambaIC/output/detect_eval")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(args)
    model = YOLO(model_path)

    runs = [
        ("original", args.original),
        ("jpeg", args.jpeg),
        ("mambaic", args.mambaic),
    ]

    all_stats = []
    for run_name, img_dir in runs:
        stats = run_on_dir(model, img_dir, args, run_name, out_dir)
        all_stats.append(stats)
        print_summary(stats)

        stats_path = out_dir / f"{run_name}_stats.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)


if __name__ == "__main__":
    main()
