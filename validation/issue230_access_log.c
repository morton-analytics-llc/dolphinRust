/*
 * Issue #230 receipt tool: log every path a process opens.
 *
 * Build:  clang -dynamiclib -O2 -o access_log.dylib validation/issue230_access_log.c
 * Use:    ACCESS_LOG=/path/to/opens.log DYLD_INSERT_LIBRARIES=$PWD/access_log.dylib dolphin ...
 *
 * Interposes open, openat and fopen (the calls Rust std, GDAL and HDF5 use
 * to open files) and appends one "<call>\t<path>" line per call. Calls made
 * from inside this library go to the originals, so logging cannot recurse.
 */
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DYLD_INTERPOSE(replacement, replacee)                                  \
    __attribute__((used)) static struct {                                     \
        const void *replacement;                                               \
        const void *replacee;                                                  \
    } interpose_##replacee __attribute__((section("__DATA,__interpose"))) = { \
        (const void *)(unsigned long)&replacement,                             \
        (const void *)(unsigned long)&replacee}

static void log_path(const char *call, const char *path) {
    static int fd = -1;
    if (fd < 0) {
        const char *target = getenv("ACCESS_LOG");
        if (target == NULL) {
            return;
        }
        fd = open(target, O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd < 0) {
            return;
        }
    }
    char line[4096];
    int n = snprintf(line, sizeof line, "%s\t%s\n", call, path ? path : "(null)");
    if (n > 0) {
        write(fd, line, (size_t)(n < (int)sizeof line ? n : (int)sizeof line - 1));
    }
}

static int logged_open(const char *path, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list args;
        va_start(args, flags);
        mode = (mode_t)va_arg(args, int);
        va_end(args);
    }
    log_path("open", path);
    return open(path, flags, mode);
}

static int logged_openat(int dirfd, const char *path, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list args;
        va_start(args, flags);
        mode = (mode_t)va_arg(args, int);
        va_end(args);
    }
    log_path("openat", path);
    return openat(dirfd, path, flags, mode);
}

static FILE *logged_fopen(const char *path, const char *mode) {
    log_path("fopen", path);
    return fopen(path, mode);
}

DYLD_INTERPOSE(logged_open, open);
DYLD_INTERPOSE(logged_openat, openat);
DYLD_INTERPOSE(logged_fopen, fopen);
