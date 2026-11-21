"""Build paired sentence fixtures, an untimed warmup, and registered timing plans."""
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent / "plans"

TEXTS = {
    "kokoro": (
        ("The harbor is quiet.", "Please wait here.", "The station lights are on.",
         "We found the missing key.", "Close the gate before leaving.", "I brought a warm jacket.",
         "Can you open the window?", "Someone called this morning.", "Your tea is ready.",
         "It started raining again.", "The post office closes soon.", "A small boat crossed the river."),
        ("I'll bring the maps for tomorrow's walk.", "The meeting starts just after nine.",
         "Leave a note beside the front door.", "We can check the details on the way.",
         "The next bus stops at the library.", "Give me a minute to look for my glasses.",
         "Someone should tell the driver where we're going.", "This route takes us past the market.",
         "I wanted to finish before everyone arrived.", "Please take your shoes off at the door."),
        ("The porch faces east.", "We should see the sunrise from here."),
    ),
    "vieneu": (
        ("Bến cảng thật yên tĩnh.", "Hãy đợi ở đây.", "Đèn nhà ga đã sáng.",
         "Chúng tôi đã tìm thấy chìa khóa.", "Nhớ đóng cổng trước khi đi.", "Tôi mang theo một chiếc áo ấm.",
         "Bạn mở cửa sổ được không?", "Sáng nay có người gọi điện.", "Trà của bạn đã pha xong.",
         "Trời lại bắt đầu mưa.", "Bưu điện sắp đóng cửa.", "Một chiếc thuyền nhỏ đi qua sông."),
        ("Tôi sẽ mang bản đồ cho chuyến đi ngày mai.", "Cuộc họp bắt đầu ngay sau chín giờ.",
         "Hãy để lại lời nhắn bên cạnh cửa chính.", "Chúng ta có thể kiểm tra lại trên đường đi.",
         "Chuyến xe buýt tiếp theo dừng ở thư viện.", "Chờ tôi một phút để tìm kính.",
         "Hãy nói cho tài xế biết chúng ta sẽ đi đâu.", "Tuyến đường này đi ngang qua chợ.",
         "Tôi muốn làm xong trước khi mọi người đến.", "Vui lòng cởi giày ở ngay cửa."),
        ("Hiên nhà hướng về phía đông.", "Chúng ta có thể ngắm bình minh từ đây."),
    ),
}


def main():
    ROOT.mkdir(exist_ok=True)
    for family, (first, second, warmup) in TEXTS.items():
        # Identical deterministic pairing and shuffle for each model's own text.
        pairs = [(a, b) for a in first for b in second]
        random.Random(61118).shuffle(pairs)
        ids = [f"{family}-{i:03}" for i in range(len(pairs))]
        rows = [(key, *pair) for key, pair in zip(ids, pairs)]
        rows.append(("warmup", *warmup))
        fixtures = ROOT / f"{family}.tsv"
        fixtures.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")
        digest = hashlib.sha256(fixtures.read_bytes()).hexdigest()
        for mode in ("stream", "full"):
            plan = {
                "levels": [1, 2, 3, 4],
                "rounds": 1,
                "waves": 5,
                "period_ms": 3000,
                "sample_rate": 24000 if family == "kokoro" else 48000,
                "mode": mode,
                "chunks_per_request": 2 if mode == "stream" else 1,
                "fixture_ids": ids[:20],
                "fixture_sha256": digest,
            }
            path = ROOT / f"screen-{family}-{mode}.json"
            path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
