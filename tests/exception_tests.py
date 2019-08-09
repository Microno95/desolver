from __future__ import absolute_import, division, print_function, unicode_literals

import desolver as de

try:
    raise de.exceptiontypes.RecursionError()
except de.exceptiontypes.RecursionError:
    pass
except:
    raise
    
try:
    raise de.exceptiontypes.FailedIntegrationError()
except de.exceptiontypes.FailedIntegrationError:
    pass
except:
    raise