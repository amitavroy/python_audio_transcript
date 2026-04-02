from typing import Any, Dict, Iterable, List


Segment = Dict[str, Any]


def normalize_and_offset_segments(chunk_segments: Iterable[Segment], *, chunk_start_seconds: float) -> List[Segment]:
    normalized: List[Segment] = []
    for seg in chunk_segments:
        raw_start = seg.get("Start", seg.get("start_time", 0.0))
        raw_end = seg.get("End", seg.get("end_time", raw_start))
        seg_start = float(raw_start)
        seg_end = float(raw_end)
        absolute_start = round(chunk_start_seconds + seg_start, 2)
        absolute_end = round(chunk_start_seconds + seg_end, 2)

        seg["Start"] = absolute_start
        seg["End"] = absolute_end
        seg["start_time"] = absolute_start
        seg["end_time"] = absolute_end

        if "Content" not in seg and "text" in seg:
            seg["Content"] = seg["text"]
        if "text" not in seg and "Content" in seg:
            seg["text"] = seg["Content"]

        normalized.append(seg)
    return normalized


def dedupe_overlap_segments(segments: Iterable[Segment]) -> List[Segment]:
    deduped: List[Segment] = []
    for current in segments:
        if deduped:
            previous = deduped[-1]
            prev_text = str(previous.get("Content") or previous.get("text") or "").strip()
            curr_text = str(current.get("Content") or current.get("text") or "").strip()
            same_text = prev_text == curr_text and bool(curr_text)
            prev_start = float(previous.get("Start", previous.get("start_time", 0.0)))
            curr_start = float(current.get("Start", current.get("start_time", 0.0)))
            near_boundary = abs(curr_start - prev_start) < 4.0
            if same_text and near_boundary:
                continue
        deduped.append(current)
    return deduped

