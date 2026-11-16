#include <stdio.h>

int main(int argc, char **argv) {
    printf("hello from pycc: sizeof(long) = %zu, sizeof(void *) = %zu\n",
           sizeof(long), sizeof(void *));
    for (int i = 1; i < argc; i++) {
        printf("argv[%d] = %s\n", i, argv[i]);
    }
    return 0;
}
