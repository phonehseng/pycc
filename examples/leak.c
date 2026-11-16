#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *copy(const char *text) {
    char *out = malloc(strlen(text) + 1);
    strcpy(out, text);
    return out;
}

int main(void) {
    char *a = copy("kept");
    char *b = copy("forgotten");
    printf("%s %s\n", a, b);
    free(a);                /* b is never freed */
    return 0;
}
