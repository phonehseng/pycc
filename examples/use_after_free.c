#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct node {
    int value;
    struct node *next;
};

int main(void) {
    struct node *head = malloc(sizeof *head);
    head->value = 42;
    head->next = NULL;
    free(head);
    printf("%d\n", head->value);    /* reads freed memory */
    return 0;
}
