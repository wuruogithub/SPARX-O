from .pastis import PASTISReader
from .pastis_pixelset import PASTISPixelSetReader

# These readers have independent optional dependencies.  Importing them in one
# shared try block made an unrelated TimeMatch/T31TFM import failure silently
# unregister MTLCCReader even when the MTLCC implementation imported correctly.
try:
    from .timematch_pixelset_reader import TimeMatchReader
except ImportError:
    TimeMatchReader = None

try:
    from .t31tfm_1618 import T31TFMReader
except ImportError:
    T31TFMReader = None

try:
    from .t31tfm_pixelset_1618 import T31TFMPixelSetReader
except ImportError:
    T31TFMPixelSetReader = None

from .mtlcc import MTLCCReader
