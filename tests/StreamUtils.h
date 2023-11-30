#ifdef _WIN32
#include <io.h>
#define NULL_DEVICE "NUL:"
#define Close _close
#define Dup _dup
#define Dup2 _dup2
#define Fileno _fileno
#else
#include <unistd.h>
#define NULL_DEVICE "/dev/null"
#define Close close
#define Dup dup
#define Dup2 dup2
#define Fileno fileno
#endif

#include <stdio.h>

namespace ont::test_utils::streams {

inline int suppress_stderr() {
    fflush(stderr);
    int fd = Dup(Fileno(stderr));
    freopen(NULL_DEVICE, "w", stderr);
    return fd;
}

inline void restore_stderr(int fd) {
    fflush(stderr);
    Dup2(fd, Fileno(stderr));
    Close(fd);
}

inline int suppress_stdout() {
    fflush(stdout);
    int fd = Dup(Fileno(stdout));
    freopen(NULL_DEVICE, "w", stdout);
    return fd;
}

inline void restore_stdout(int fd) {
    fflush(stdout);
    Dup2(fd, Fileno(stdout));
    Close(fd);
}

}  // namespace ont::test_utils::streams
