from collections import defaultdict

def accumulate_disp_dict(val_disp_dicts):
    """
    val_disp_dicts: list of dicts, 每个 dict 的值是标量 tensor
    返回一个新的 dict，值是平均后的 float
    """
    total = defaultdict(float)
    count = len(val_disp_dicts)

    for d in val_disp_dicts:
        for k, v in d.items():
            total[k] += v.item()  # tensor -> float

    avg_dict = {k: total[k] / count for k in total}
    return avg_dict