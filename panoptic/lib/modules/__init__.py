from .GP_block import *
from .temp_spat_transformer import TempSpatTransformer
from .transformer import *
from .pse import PixelSetEncoder
from .ltae import LTAE
from .unet import UNet
from .exchanger import Exchanger
from .calendar_phase_ssm import CalendarPhaseTemporalEncoder

# These heads are unrelated to parcel classification and U-Net semantic
# segmentation.  Keep them optional so the clean TU.2 experiment does not
# require Detectron2 or torch-scatter merely to import the model package.
try:
    from .fpn import FPN
    from .pvt import PVT2
    from .swin import SwinTransformer
    from .maskformer_model import MaskFormer
except ImportError:
    FPN = None
    PVT2 = None
    SwinTransformer = None
    MaskFormer = None

try:
    from .paps_orig import PaPs
except ImportError:
    PaPs = None
