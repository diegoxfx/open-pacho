from openpacho.opencl_backend.opencl_context import OpenCLContextManager

def pick_optimal_group_size_vector_dot(length):
    """
    Heuristically pick a local work-group size for a dot-product kernel,
    given hardware constraints and the problem size.

    Args:

    Returns:
        int: A heuristic local work-group size aligned with the device constraints.
    """
    device = OpenCLContextManager().device
    pwgsm = device.preferred_work_group_size_multiple
    mwgs = device.max_work_group_size
    mcu = device.max_compute_units

    valid_sizes = [8, 16, 32, 64, 128, 256, 512]

    # 1) Clamp by device's max_work_group_size:
    valid_sizes = [sz for sz in valid_sizes if sz <= mwgs]

    # 2) Further restrict to those that exactly divide 'length'
    #    (to ensure length % group_size == 0).
    valid_sizes = [sz for sz in valid_sizes if length % sz == 0]

    # If this becomes empty, we have no valid group size. 
    # (Should not happen if 'length' is at least a multiple of 8
    #  and 'max_work_group_size' >= 8, but let's be safe.)
    if not valid_sizes:
        # fallback: pick 8 if it's <= max_work_group_size and divides length, 
        # or just pick 1 if you must handle an edge case.
        # But since you guarantee length is multiple of 8, let's default to 8 if possible:
        fallback = 8 if (8 <= mwgs and length % 8 == 0) else 1
        return fallback

    # --------------------------------------------------------------------
    # Next, we follow the previous heuristic to pick the "best" from valid_sizes.
    # We'll compute a target based on distributing length across max_compute_units.
    # --------------------------------------------------------------------

    # If length < 8, still pick 8 (though you said length >= multiple of 8, so rarely triggered).
    if length < 8:
        return 8

    # "Target" tries to balance how many threads to run per compute unit
    target = length // mcu

    # Ensure target is within [8, 512] (and <= device max), 
    # though we've already filtered out invalid group sizes above.
    target = max(8, min(target, 512, mwgs))

    # Sort valid_sizes descending so we can pick the largest that meets conditions.
    valid_sizes_desc = sorted(valid_sizes, reverse=True)

    # 1) Try to find the largest that is <= target AND a multiple of preferred_work_group_size_multiple
    best_candidate = None
    for sz in valid_sizes_desc:
        if sz <= target and (sz % pwgsm == 0):
            best_candidate = sz
            break

    # 2) If none found, pick the largest <= target
    if best_candidate is None:
        for sz in valid_sizes_desc:
            if sz <= target:
                best_candidate = sz
                break

    # 3) Ensure we don't pick a group size bigger than the length (unlikely if length is huge),
    #    but let's be consistent with the old logic:
    if best_candidate is not None and best_candidate > length:
        # fallback to the largest valid size <= length
        fallback = None
        for sz in valid_sizes_desc:
            if sz <= length:
                fallback = sz
                break
        if fallback is not None:
            best_candidate = fallback

    # 4) If still none, pick the smallest from valid_sizes (very unlikely here).
    if best_candidate is None:
        best_candidate = valid_sizes[-1]  # the smallest after sort is last

    return best_candidate