""" Estimate computation and communication runtimes by direct execution

See also:
    communication       - module for network communication
    commtime_dict       - converts network discription to timing dict
    make_commtime_function - converts timing dict to callable function

    computation         - module for computation times
    comptime_dict       - converts fgraphs and list of machines to timing dicts
    make_runtime_function - converts timing dict to callable function
"""

import communication
import computation
import util

from communication import commtime_dict, make_commtime_function
from computation   import comptime_dict, make_runtime_function
