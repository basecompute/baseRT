#ifndef CBASERT_SHIM_H
#define CBASERT_SHIM_H

// Umbrella header for the CBaseRT clang module. baseRT.h pulls in types.h.
// These headers are vendored copies of include/baseRT/{baseRT,types}.h so the
// Swift package is self-contained; keep them in sync when the C API changes.
#include "baseRT.h"

#endif /* CBASERT_SHIM_H */
