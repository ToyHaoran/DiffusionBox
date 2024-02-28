from mega_core import _C

from torch.cuda.amp import autocast,GradScaler
# from apex import amp

# Only valid with fp32 inputs - give AMP the hint
# fps = amp.float_function(_C.furthest_point_sampling)  # 注释掉amp
# 使用with autocast():代替。
