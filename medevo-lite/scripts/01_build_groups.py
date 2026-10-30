"""Step 2: define tag groups -> data/groups.json.

One group per value of modality, location and content_type. A value with fewer
than MIN_GROUP feedback (val) questions is merged into "<axis>:other", which is
how the plan's "back off to the axis-level total" is implemented: the merged
group gets one error rate and one quota, and keeps its own pool questions.
"""
import collections

from common import AXES, DATA, MIN_GROUP, load_split, write_json


def main():
    val = load_split("val")
    backoff, sizes = {}, {}
    for axis in AXES:
        counts = collections.Counter(r[axis] for r in val)
        for value, n in counts.items():
            gid = f"{axis}:{value}"
            if n < MIN_GROUP:
                backoff[gid] = f"{axis}:other"
        for value, n in counts.items():
            gid = backoff.get(f"{axis}:{value}", f"{axis}:{value}")
            sizes[gid] = sizes.get(gid, 0) + n

    groups = {
        "axes": list(AXES),
        "min_group": MIN_GROUP,
        "backoff": backoff,
        "groups": sorted(sizes),
        "val_sizes": sizes,
    }
    write_json(DATA / "groups.json", groups)
    for gid in groups["groups"]:
        print(f"{gid:35s} n_val={sizes[gid]}")
    if backoff:
        print("backed off:", ", ".join(f"{k} -> {v}" for k, v in sorted(backoff.items())))


if __name__ == "__main__":
    main()
