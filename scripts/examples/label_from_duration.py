"""Example ad-hoc pipeline script.

Every script owns its business logging by calling ``context.log``.  The engine
adds timestamps, step/sample identifiers and safely appends JSONL events.
"""


def process(sample, params, context):
    minimum = float(params.get("minimum_seconds", 1.0))
    duration = float(sample.get("duration") or 0.0)
    accepted = duration >= minimum
    context.log("duration evaluated", duration=duration, minimum=minimum, accepted=accepted)
    return {
        "labels": {"duration_pass": accepted},
        "quality": {"duration_threshold": minimum},
    }
