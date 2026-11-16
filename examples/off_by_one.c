#include <stdio.h>

int main(void) {
    int scores[5] = {90, 85, 77, 68, 99};
    int total = 0;
    for (int i = 0; i <= 5; i++) {     /* one step too far */
        total += scores[i];
    }
    printf("total = %d\n", total);
    return 0;
}
