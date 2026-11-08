/* C++ runtime pieces TFLite Micro references; it never allocates or deletes. */
void __cxa_pure_virtual(void) { while (1); }
void _ZdlPv(void *p) { (void)p; }
void _ZdlPvj(void *p, unsigned n) { (void)p; (void)n; }
void *__dso_handle = 0;
#include <stdarg.h>
#include <stdio.h>
#include <stddef.h>
void __assert_func(const char *file, int line, const char *func, const char *expr) {
    printf("FAIL: assert %s at %s:%d\n", expr, file, line);
    while (1);
}
/* TFLite Micro's log hooks (its own debug_log.cc writes to stderr, which this runtime lacks) */
void DebugLog(const char *format, va_list args) { vprintf(format, args); }
int DebugVsnprintf(char *buffer, size_t buf_size, const char *format, va_list vlist) { return vsnprintf(buffer, buf_size, format, vlist); }
