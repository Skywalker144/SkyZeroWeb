#include <cstdlib>
extern "C" [[noreturn]] void __wrap_quick_exit(int status) { std::_Exit(status); }
