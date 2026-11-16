"""Review scenarios: typical teaching-course C programs, run end to end.

Each test types a real program into the project, builds it with ``cc`` (or ``make``)
and runs it through the headless terminal, exactly as an interactive terminal would. The second half executes claims made by
``docs/SUPPORTED_C.md`` so the page cannot drift away from what the toolchain does.

Expected outputs were worked out by hand from the C standard (and agree with gcc on
x86-64 Linux), never copied from pycc's own output.
"""
from __future__ import annotations

from tests.integration.conftest import Terminal

BUILD = "cc -Wall -std=c99 main.c -o prog"


def build_and_run(term: Terminal, source: str, run: str = "./prog") -> str:
    """Compile ``source`` as main.c without any diagnostic, run it and return stdout."""
    term.write({"main.c": source})
    built = term.run(BUILD)
    assert built.exit_code == 0 and built.stderr == "", built.stderr
    return term.ok(run).stdout


# --- week 3: strings -------------------------------------------------------------------

def test_reverse_in_place_and_palindromes(term: Terminal) -> None:
    source = """\
#include <stdio.h>
#include <string.h>
#include <ctype.h>

void reverse(char *s) {
    size_t i, n = strlen(s);
    for (i = 0; i < n / 2; i++) {
        char tmp = s[i];
        s[i] = s[n - 1 - i];
        s[n - 1 - i] = tmp;
    }
}

int is_palindrome(const char *s) {
    const char *end = s + strlen(s) - 1;
    while (s < end) {
        if (tolower((unsigned char)*s) != tolower((unsigned char)*end)) {
            return 0;
        }
        s++;
        end--;
    }
    return 1;
}

int main(void) {
    char word[] = "harvest";
    const char *tests[] = { "Level", "wheat", "noon" };
    int i;
    reverse(word);
    printf("%s\\n", word);
    for (i = 0; i < 3; i++) {
        printf("%s: %s\\n", tests[i], is_palindrome(tests[i]) ? "yes" : "no");
    }
    return 0;
}
"""
    assert build_and_run(term, source) == "tsevrah\nLevel: yes\nwheat: no\nnoon: yes\n"


def test_caesar_cipher_from_argv(term: Terminal) -> None:
    source = """\
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

int main(int argc, char *argv[]) {
    int shift, i;
    char *p;
    if (argc < 3) {
        fprintf(stderr, "usage: %s SHIFT WORD...\\n", argv[0]);
        return 2;
    }
    shift = atoi(argv[1]);
    for (i = 2; i < argc; i++) {
        for (p = argv[i]; *p; p++) {
            if (isupper(*p)) {
                *p = 'A' + (*p - 'A' + shift) % 26;
            } else if (islower(*p)) {
                *p = 'a' + (*p - 'a' + shift) % 26;
            }
        }
        printf("%s%c", argv[i], i == argc - 1 ? '\\n' : ' ');
    }
    return 0;
}
"""
    assert build_and_run(term, source, "./prog 3 Hello, Zebra") == "Khoor, Cheud\n"
    usage = term.run("./prog")
    assert usage.exit_code == 2 and usage.stderr == "usage: ./prog SHIFT WORD...\n"


def test_word_count_reads_a_pipe_with_fgets(term: Terminal) -> None:
    source = """\
#include <stdio.h>
#include <ctype.h>

int main(void) {
    char line[64];
    int lines = 0, words = 0;
    while (fgets(line, sizeof line, stdin) != NULL) {
        int in_word = 0;
        char *p;
        lines++;
        for (p = line; *p != '\\0'; p++) {
            if (isspace((unsigned char)*p)) {
                in_word = 0;
            } else if (!in_word) {
                in_word = 1;
                words++;
            }
        }
    }
    printf("%d %d\\n", lines, words);
    return 0;
}
"""
    term.write({"notes.txt": "wheat grows fast\ncarrots slow\n\npumpkins need water\n"})
    assert build_and_run(term, source, "cat notes.txt | ./prog") == "4 8\n"
    assert term.ok("./prog < notes.txt").stdout == "4 8\n"


def test_strtok_splits_a_csv_line_and_sums_it(term: Terminal) -> None:
    source = """\
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    char line[] = "wheat,12,carrot,7,pumpkin,3";
    char *token = strtok(line, ",");
    int total = 0;
    while (token != NULL) {
        char *count = strtok(NULL, ",");
        printf("%-8s|%3d|\\n", token, atoi(count));
        total += atoi(count);
        token = strtok(NULL, ",");
    }
    printf("total=%d\\n", total);
    return 0;
}
"""
    assert build_and_run(term, source) == (
        "wheat   | 12|\ncarrot  |  7|\npumpkin |  3|\ntotal=22\n")


def test_string_library_builds_a_report(term: Terminal) -> None:
    source = """\
#include <stdio.h>
#include <string.h>
#include <ctype.h>

int main(void) {
    char name[16];
    char report[64] = "crop: ";
    size_t i;
    strncpy(name, "pumpkin", sizeof name - 1);
    name[sizeof name - 1] = '\\0';
    for (i = 0; name[i]; i++) {
        name[i] = (char)toupper((unsigned char)name[i]);
    }
    strcat(report, name);
    printf("%s (%zu chars)\\n", report, strlen(report));
    printf("%d %d %d\\n", strcmp("apple", "apple") == 0, strcmp("a", "b") < 0,
           strncmp("wheatfield", "wheat", 5) == 0);
    printf("[%s]\\n", strchr(report, 'P'));
    printf("[%s]\\n", strstr("the wheat field", "field"));
    return 0;
}
"""
    assert build_and_run(term, source) == (
        "crop: PUMPKIN (13 chars)\n1 1 1\n[PUMPKIN]\n[field]\n")


# --- week 4-5: structs and pointers ----------------------------------------------------

def test_struct_pointers_update_a_drone_record(term: Terminal) -> None:
    source = """\
#include <stdio.h>

struct Point { int x, y; };

struct Drone {
    char name[12];
    struct Point pos;
    int fuel;
};

static const int DX[4] = { 0, 1, 0, -1 };
static const int DY[4] = { 1, 0, -1, 0 };

int fly(struct Drone *d, int heading, int steps) {
    int done = 0;
    while (done < steps && d->fuel > 0) {
        d->pos.x += DX[heading];
        d->pos.y += DY[heading];
        d->fuel--;
        done++;
    }
    return done;
}

void show(const struct Drone *d) {
    printf("%s at (%d,%d) fuel %d\\n", d->name, d->pos.x, d->pos.y, d->fuel);
}

int main(void) {
    struct Drone fleet[2] = { { "Rover", { 0, 0 }, 5 }, { "Cog", { 3, 3 }, 2 } };
    struct Drone *d;
    fly(&fleet[0], 1, 3);
    fly(&fleet[1], 2, 4);
    for (d = fleet; d < fleet + 2; d++) {
        show(d);
    }
    printf("%d\\n", (int)sizeof(struct Drone));
    return 0;
}
"""
    assert build_and_run(term, source) == (
        "Rover at (3,0) fuel 2\nCog at (3,1) fuel 0\n24\n")


def test_swap_through_pointers_and_pointer_to_pointer(term: Terminal) -> None:
    source = """\
#include <stdio.h>

void swap_int(int *a, int *b) { int t = *a; *a = *b; *b = t; }

void swap_str(const char **a, const char **b) {
    const char *t = *a;
    *a = *b;
    *b = t;
}

int main(void) {
    int x = 1, y = 2;
    const char *first = "wheat", *second = "carrot";
    int values[] = { 5, 3, 8, 1 };
    int n = sizeof values / sizeof values[0];
    int i, j;
    swap_int(&x, &y);
    swap_str(&first, &second);
    for (i = 0; i < n - 1; i++) {
        for (j = 0; j < n - 1 - i; j++) {
            if (values[j] > values[j + 1]) {
                swap_int(&values[j], values + j + 1);
            }
        }
    }
    printf("%d %d %s %s\\n", x, y, first, second);
    for (i = 0; i < n; i++) {
        printf("%d%s", values[i], i < n - 1 ? "," : "\\n");
    }
    return 0;
}
"""
    assert build_and_run(term, source) == "2 1 carrot wheat\n1,3,5,8\n"


def test_qsort_orders_structs_with_a_comparator(term: Terminal) -> None:
    source = """\
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *crop;
    int count;
} Tally;

static int by_count_desc(const void *a, const void *b) {
    const Tally *ta = a;
    const Tally *tb = b;
    if (ta->count != tb->count) {
        return tb->count - ta->count;
    }
    return strcmp(ta->crop, tb->crop);
}

int main(void) {
    Tally t[] = { { "wheat", 4 }, { "carrot", 9 }, { "beet", 4 }, { "pumpkin", 1 } };
    size_t i, n = sizeof t / sizeof t[0];
    qsort(t, n, sizeof t[0], by_count_desc);
    for (i = 0; i < n; i++) {
        printf("%zu. %s %d\\n", i + 1, t[i].crop, t[i].count);
    }
    return 0;
}
"""
    assert build_and_run(term, source) == (
        "1. carrot 9\n2. beet 4\n3. wheat 4\n4. pumpkin 1\n")


def test_switch_over_an_enum_with_fall_through(term: Terminal) -> None:
    source = """\
#include <stdio.h>

enum Season { SPRING, SUMMER, AUTUMN, WINTER };

const char *name(enum Season s) {
    switch (s) {
    case SPRING: return "spring";
    case SUMMER: return "summer";
    case AUTUMN: return "autumn";
    default:     return "winter";
    }
}

int work_days(enum Season s) {
    int days = 0;
    switch (s) {
    case SPRING:
        days += 10;
        /* fall through: spring also has the summer work */
    case SUMMER:
        days += 20;
        break;
    case AUTUMN:
        days = 15;
        break;
    default:
        break;
    }
    return days;
}

int main(void) {
    enum Season s;
    for (s = SPRING; s <= WINTER; s++) {
        printf("%s %d\\n", name(s), work_days(s));
    }
    return 0;
}
"""
    assert build_and_run(term, source) == "spring 30\nsummer 20\nautumn 15\nwinter 0\n"


def test_recursion_and_unsigned_long_arithmetic(term: Terminal) -> None:
    source = """\
#include <stdio.h>

unsigned long factorial(unsigned n) {
    return n <= 1 ? 1UL : n * factorial(n - 1);
}

int fib(int n) {
    return n < 2 ? n : fib(n - 1) + fib(n - 2);
}

int main(void) {
    printf("%lu\\n", factorial(20));
    printf("%d\\n", fib(15));
    printf("%d %d %d\\n", -7 / 2, -7 % 2, 7 / -2);
    printf("%u\\n", (unsigned)-1);
    printf("0x%08x\\n", 0xF0u | (1u << 3));
    return 0;
}
"""
    assert build_and_run(term, source) == (
        "2432902008176640000\n610\n-3 -1 -3\n4294967295\n0x000000f8\n")


# --- week 6: malloc'd data -------------------------------------------------------------

LIST_PROGRAM = """\
#include <stdio.h>
#include <stdlib.h>

struct Node {
    int value;
    struct Node *next;
};

struct Node *push(struct Node *head, int value) {
    struct Node *node = malloc(sizeof *node);
    if (node == NULL) {
        return head;
    }
    node->value = value;
    node->next = head;
    return node;
}

struct Node *reverse(struct Node *head) {
    struct Node *prev = NULL;
    while (head != NULL) {
        struct Node *next = head->next;
        head->next = prev;
        prev = head;
        head = next;
    }
    return prev;
}

void print(const struct Node *n) {
    for (; n != NULL; n = n->next) {
        printf("%d%s", n->value, n->next ? " -> " : "\\n");
    }
}

void free_list(struct Node *n) {
    while (n != NULL) {
        struct Node *next = n->next;
        free(n);
        n = next;
    }
}

int main(void) {
    struct Node *list = NULL;
    int i;
    for (i = 1; i <= 4; i++) {
        list = push(list, i * i);
    }
    print(list);
    list = reverse(list);
    print(list);
    FREE_CALL
    return 0;
}
"""


def test_linked_list_push_reverse_free_without_leaks(term: Terminal) -> None:
    out = build_and_run(term, LIST_PROGRAM.replace("FREE_CALL", "free_list(list);"))
    assert out == "16 -> 9 -> 4 -> 1\n1 -> 4 -> 9 -> 16\n"
    line = term.run("./prog")
    assert line.stderr == "" and line.runs[0].leaks == 0


def test_linked_list_that_forgets_free_reports_each_leak(term: Terminal) -> None:
    build_and_run(term, LIST_PROGRAM.replace("FREE_CALL", ""))
    line = term.run("./prog")
    assert line.exit_code == 0
    assert line.runs[0].leaks == 4
    assert line.stderr.count("16 bytes in 1 blocks are definitely lost") == 4


def test_realloc_grows_a_dynamic_array(term: Terminal) -> None:
    source = """\
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    size_t cap = 2, len = 0, i;
    int *data = malloc(cap * sizeof *data);
    int sum = 0;
    for (i = 0; i < 10; i++) {
        if (len == cap) {
            int *bigger = realloc(data, 2 * cap * sizeof *data);
            if (bigger == NULL) {
                free(data);
                return 1;
            }
            data = bigger;
            cap *= 2;
        }
        data[len++] = (int)(i * 3);
    }
    for (i = 0; i < len; i++) {
        sum += data[i];
    }
    printf("len=%zu cap=%zu sum=%d\\n", len, cap, sum);
    free(data);
    return 0;
}
"""
    assert build_and_run(term, source) == "len=10 cap=16 sum=135\n"


def test_strdup_and_calloc_build_a_string_table(term: Terminal) -> None:
    source = """\
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const char *crops[] = { "wheat", "carrot", "pumpkin" };
    char **copies = calloc(3, sizeof(char *));
    int i;
    for (i = 0; i < 3; i++) {
        copies[i] = strdup(crops[i]);
        copies[i][0] = (char)(copies[i][0] - 'a' + 'A');
    }
    for (i = 0; i < 3; i++) {
        printf("%s/%s\\n", crops[i], copies[i]);
        free(copies[i]);
    }
    free(copies);
    return 0;
}
"""
    assert build_and_run(term, source) == "wheat/Wheat\ncarrot/Carrot\npumpkin/Pumpkin\n"


# --- week 6: files ---------------------------------------------------------------------

def test_fprintf_writes_a_file_the_shell_can_read(term: Terminal) -> None:
    source = """\
#include <stdio.h>

int main(void) {
    FILE *out = fopen("harvest.csv", "w");
    FILE *in;
    int day, count, total = 0;
    if (out == NULL) {
        perror("harvest.csv");
        return 1;
    }
    for (day = 1; day <= 3; day++) {
        fprintf(out, "%d,%d\\n", day, day * 4);
    }
    fclose(out);
    in = fopen("harvest.csv", "r");
    while (fscanf(in, "%d,%d", &day, &count) == 2) {
        total += count;
    }
    fclose(in);
    printf("total %d\\n", total);
    return 0;
}
"""
    assert build_and_run(term, source) == "total 24\n"
    assert term.ok("cat harvest.csv").stdout == "1,4\n2,8\n3,12\n"
    assert term.ok("wc -l < harvest.csv").stdout.strip() == "3"


def test_scanf_reads_numbers_until_end_of_file(term: Terminal) -> None:
    source = """\
#include <stdio.h>

int main(void) {
    int n, count = 0, best = 0;
    double sum = 0.0;
    while (scanf("%d", &n) == 1) {
        sum += n;
        if (count == 0 || n > best) {
            best = n;
        }
        count++;
    }
    if (count > 0) {
        printf("n=%d max=%d mean=%.2f\\n", count, best, sum / count);
    }
    return 0;
}
"""
    term.write({"nums.txt": "4 8\n15\n16 23 42\n"})
    assert build_and_run(term, source, "./prog < nums.txt") == "n=6 max=42 mean=18.00\n"
    assert term.ok("seq 1 5 | ./prog").stdout == "n=5 max=5 mean=3.00\n"


# --- week 7: a four-file project built with make ---------------------------------------

PROJECT_FILES = {
    "field.h": """\
#ifndef FIELD_H
#define FIELD_H

struct Field {
    int width;
    int height;
    int *counts;
};

struct Field *field_new(int width, int height);
void field_free(struct Field *f);
int *field_cell(struct Field *f, int x, int y);

#endif
""",
    "field.c": """\
#include <stdlib.h>
#include "field.h"

struct Field *field_new(int width, int height) {
    struct Field *f = malloc(sizeof *f);
    f->width = width;
    f->height = height;
    f->counts = calloc((size_t)(width * height), sizeof(int));
    return f;
}

void field_free(struct Field *f) {
    free(f->counts);
    free(f);
}

int *field_cell(struct Field *f, int x, int y) {
    return &f->counts[y * f->width + x];
}
""",
    "survey.h": """\
#ifndef SURVEY_H
#define SURVEY_H

#include "field.h"

extern int surveys_done;
int survey(struct Field *f);

#endif
""",
    "survey.c": """\
#include <stdio.h>
#include "survey.h"

int surveys_done = 0;

int survey(struct Field *f) {
    int x, y, hits = 0;
    for (y = 0; y < f->height; y++) {
        for (x = 0; x < f->width; x++) {
            *field_cell(f, x, y) = (x + y) % 3;
            if (*field_cell(f, x, y) == 0) {
                hits++;
            }
        }
    }
    surveys_done++;
    return hits;
}
""",
    "main.c": """\
#include <stdio.h>
#include "field.h"
#include "survey.h"

int main(void) {
    struct Field *f = field_new(4, 3);
    int hits = survey(f);
    printf("%dx%d field, %d hits, %d survey\\n", f->width, f->height, hits,
           surveys_done);
    field_free(f);
    return 0;
}
""",
    "Makefile": """\
CC = gcc
CFLAGS = -Wall -Wextra -std=c99
SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)

survey_bot: $(OBJS)
\t$(CC) $(CFLAGS) $^ -o $@

%.o: %.c field.h survey.h
\t$(CC) $(CFLAGS) -c $< -o $@

.PHONY: run clean
run: survey_bot
\t@./survey_bot

clean:
\t$(RM) $(OBJS) survey_bot
""",
}


def test_four_file_project_builds_with_make_and_runs(term: Terminal) -> None:
    term.write(PROJECT_FILES)
    echoed = term.ok("make").stdout.splitlines()
    assert sorted(echoed[:3]) == sorted(
        f"gcc -Wall -Wextra -std=c99 -c {name}.c -o {name}.o"
        for name in ("field", "main", "survey"))
    assert echoed[3] == "gcc -Wall -Wextra -std=c99 field.o main.o survey.o -o survey_bot"
    assert term.ok("make run").stdout == "4x3 field, 4 hits, 1 survey\n"
    assert term.ok("make").stdout == "make: 'survey_bot' is up to date.\n"


def test_four_file_project_rebuilds_after_touching_a_header(term: Terminal) -> None:
    term.write(PROJECT_FILES)
    term.ok("make")
    term.ok("touch survey.h")
    rebuilt = term.ok("make").stdout.splitlines()
    assert len(rebuilt) == 4                    # every .o depends on survey.h, then link
    term.ok("make clean")
    assert not term.vfs.exists("/home/user/project/survey_bot")


# --- week 7: a shell script with pipes ------------------------------------------------

def test_shell_script_with_pipes_counts_program_output(term: Terminal) -> None:
    source = """\
#include <stdio.h>

int main(int argc, char **argv) {
    int i, n = argc > 1 ? argc - 1 : 0;
    for (i = 1; i <= n; i++) {
        printf("%s %s\\n", i % 2 ? "WARN" : "ok", argv[i]);
    }
    return n > 3 ? 1 : 0;
}
"""
    script = """\
#!/bin/sh
# Build the bot, then count its warnings.
cc -Wall main.c -o prog || exit 1
for crop in wheat carrot; do
    echo "== $crop"
done
./prog a b c d e | grep WARN | wc -l
./prog a b c | sort -r | head -n 1
./prog a b c d
echo "status $?"
"""
    term.write({"main.c": source, "check.sh": script})
    denied = term.run("./check.sh")
    assert denied.exit_code == 126
    term.ok("chmod +x check.sh")
    out = term.ok("./check.sh").stdout.splitlines()
    assert out[:2] == ["== wheat", "== carrot"]
    assert out[2].strip() == "3"
    assert out[3] == "ok b"
    assert out[-1] == "status 1"


def test_pipeline_feeds_one_program_into_another(term: Terminal) -> None:
    upper = """\
#include <stdio.h>
#include <ctype.h>

int main(void) {
    int c;
    while ((c = getchar()) != EOF) {
        putchar(toupper(c));
    }
    return 0;
}
"""
    term.write({"main.c": upper})
    term.ok(BUILD)
    line = term.ok("echo 'wheat and carrots' | ./prog | ./prog > out.txt && cat out.txt")
    assert line.stdout == "WHEAT AND CARROTS\n"
    assert line.finished.constructs["pipeline"] >= 1


# --- docs/SUPPORTED_C.md: its claims, executed ----------------------------------------

def test_doc_sizes_match_the_lp64_table(term: Terminal) -> None:
    source = """\
#include <stdio.h>
#include <stdbool.h>

struct Mixed { char c; int i; char d; double x; };

int main(void) {
    printf("%zu %zu %zu %zu %zu %zu %zu %zu %zu %zu\\n", sizeof(char), sizeof(short),
           sizeof(int), sizeof(long), sizeof(long long), sizeof(float), sizeof(double),
           sizeof(void *), sizeof(bool), sizeof(struct Mixed));
    return 0;
}
"""
    assert build_and_run(term, source) == "1 2 4 8 8 4 8 8 1 24\n"


def test_doc_rand_is_the_standard_sample_generator_seeded_with_1(term: Terminal) -> None:
    source = """\
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int a = rand();
    int b = rand();
    srand(1);
    printf("%d %d %d %d\\n", a, b, rand(), RAND_MAX);
    return 0;
}
"""
    # next = next * 1103515245 + 12345; return (next / 65536) % 32768, from seed 1.
    assert build_and_run(term, source) == "16838 5758 16838 32767\n"


def test_doc_printf_null_pointer_prints_nil(term: Terminal) -> None:
    source = """\
#include <stdio.h>

int main(void) {
    printf("%p|%5.1f|%-4d|%+d|%x|%o|%e|%c%%\\n", (void *)0, 3.14159, 7, 5, 255, 8,
           1500.0, 'A');
    return 0;
}
"""
    assert build_and_run(term, source) == "(nil)|  3.1|7   |+5|ff|10|1.500000e+03|A%\n"


def test_doc_func_and_adjacent_string_literals(term: Terminal) -> None:
    source = """\
#include <stdio.h>

void report(void) {
    printf("in %s: " "joined" " literals\\n", __func__);
}

int main(void) {
    report();
}
"""
    term.write({"main.c": source})
    line = term.ok("cc main.c -o prog && ./prog")     # falling off main returns 0
    assert line.stdout == "in report: joined literals\n"


def test_doc_goto_is_refused_with_its_message(term: Terminal) -> None:
    term.write({"main.c": "int main(void) {\n    goto end;\nend:\n    return 0;\n}\n"})
    line = term.run("cc main.c -o prog")
    assert line.exit_code == 1
    assert "labels and 'goto' are not supported" in line.stderr
    assert "unsupported_feature" in line.codes


def test_doc_variable_length_array_is_refused(term: Terminal) -> None:
    term.write({"main.c": "int main(void) {\n    int n = 3;\n    int a[n];\n"
                          "    a[0] = 1;\n    return a[0];\n}\n"})
    line = term.run("cc main.c -o prog")
    assert line.exit_code == 1
    assert "size of array 'a' is not an integer constant" in line.stderr


def test_doc_stringizing_macro_is_an_error(term: Terminal) -> None:
    term.write({"main.c": "#define NAME(x) #x\nint main(void) { return 0; }\n"})
    line = term.run("cc main.c -o prog")
    assert line.exit_code == 1
    assert "the '#' operator is not supported" in line.stderr


def test_doc_unknown_standard_header_is_fatal(term: Terminal) -> None:
    term.write({"main.c": "#include <unistd.h>\nint main(void) { return 0; }\n"})
    line = term.run("cc main.c -o prog")
    assert line.exit_code == 1
    assert "fatal error: unistd.h: No such file or directory" in line.stderr
    assert "include_not_found" in line.codes


def test_doc_perror_after_a_failing_fopen(term: Terminal) -> None:
    source = """\
#include <stdio.h>
#include <errno.h>

int main(void) {
    FILE *f = fopen("plan.txt", "r");
    if (f == NULL) {
        perror("plan.txt");
        return errno == ENOENT ? 3 : 4;
    }
    fclose(f);
    return 0;
}
"""
    term.write({"main.c": source})
    term.ok(BUILD)
    line = term.run("./prog")
    assert line.exit_code == 3
    assert "plan.txt: No such file or directory" in line.stderr.splitlines()


def test_doc_system_always_returns_minus_one(term: Terminal) -> None:
    source = """\
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("%d\\n", system("ls"));
    return 0;
}
"""
    assert build_and_run(term, source) == "-1\n"


def test_doc_division_by_zero_is_status_136(term: Terminal) -> None:
    source = """\
#include <stdio.h>

int share(int baskets) { return 12 / baskets; }

int main(void) {
    printf("%d\\n", share(0));
    return 0;
}
"""
    term.write({"main.c": source})
    term.ok(BUILD)
    line = term.run("./prog")
    assert line.exit_code == 136
    assert "Floating point exception (core dumped)" in line.stderr
    assert line.runs[0].trap.code == "division_by_zero"


def test_doc_runaway_loop_hits_the_instruction_budget(term: Terminal) -> None:
    source = """\
int main(void) {
    unsigned long n = 0;
    while (1) {
        n++;
    }
    return 0;
}
"""
    term.write({"main.c": source})
    term.ok("cc main.c -o prog")
    line = term.run("./prog")
    assert line.exit_code == 137
    assert "Killed" in line.stderr
    assert line.runs[0].trap.code == "instruction_budget"


def test_doc_out_of_bounds_names_the_array(term: Terminal) -> None:
    source = """\
int main(void) {
    int field[8];
    int i;
    for (i = 0; i <= 8; i++) {
        field[i] = i;
    }
    return field[0];
}
"""
    term.write({"main.c": source})
    term.ok("cc main.c -o prog")
    line = term.run("./prog")
    assert line.exit_code == 139
    assert "index 8 is past the end of 'field' (8 elements)" in line.stderr
    assert "Segmentation fault (core dumped)" in line.stderr


def test_doc_werror_marks_warnings(term: Terminal) -> None:
    term.write({"main.c": "int main(void) {\n    int unused;\n    return 0;\n}\n"})
    line = term.run("cc -Wall -Werror main.c -o prog")
    assert line.exit_code == 1
    assert "[-Werror]" in line.stderr
    assert not term.vfs.exists("/home/user/project/prog")


def test_doc_scanset_reads_up_to_a_comma(term: Terminal) -> None:
    source = """\
#include <stdio.h>

int main(void) {
    char crop[16];
    int count = 0;
    int got = sscanf("carrot,17", "%15[^,],%d", crop, &count);
    printf("%d %s %d\\n", got, crop, count);
    return 0;
}
"""
    assert build_and_run(term, source) == "2 carrot 17\n"


def test_doc_static_is_private_to_its_file(term: Terminal) -> None:
    term.write({
        "helper.c": "static int secret(void) { return 7; }\nint open_door(void) "
                    "{ return secret(); }\n",
        "main.c": "int secret(void);\nint main(void) { return secret(); }\n",
    })
    line = term.run("cc helper.c main.c -o prog")
    assert line.exit_code == 1
    assert "undefined reference to 'secret'" in line.stderr
    assert "collect2: error: ld returned 1 exit status" in line.stderr


# --- regressions found by this review -------------------------------------------------

def test_strcpy_overflow_explains_where_the_copy_ends(term: Terminal) -> None:
    """The copy starts at index 0 (valid): the friendly line must not blame index 0."""
    source = """\
#include <string.h>

int main(void) {
    char name[8];
    strcpy(name, "Rover-the-explorer");
    return name[0];
}
"""
    term.write({"main.c": source})
    term.ok(BUILD)
    line = term.run("./prog")
    assert line.exit_code == 139
    assert ("main.c:5:5: error: writing 19 bytes at index 0 runs past the end of 'name' "
            "(8 elements)") in line.stderr
    assert "covers 19 elements, up to index 18" in line.stderr
    assert "Index 0 is outside it" not in line.stderr


def test_a_designated_initializer_is_the_only_error_reported(term: Terminal) -> None:
    """The refusal must not unwind the function and make its statements look broken."""
    source = """\
struct P { int x, y; };

int main(void) {
    struct P p = { .x = 1, .y = 2 };
    int grid[3] = { [2] = 5 };
    return p.x + grid[0];
}
"""
    term.write({"main.c": source})
    line = term.run("cc main.c -o prog")
    assert line.exit_code == 1
    assert set(line.codes) == {"unsupported_feature"}
    assert "main.c:4:20: error: designated initializers are not supported" in line.stderr
    assert "main.c:5:21: error: designated initializers are not supported" in line.stderr
    assert "expected" not in line.stderr


def test_conditional_cflags_take_effect_as_in_gnu_make(term: Terminal) -> None:
    """GNU make does not predefine CFLAGS, so ``CFLAGS ?= -Wall`` sets it; CC is predefined."""
    term.write({
        "main.c": "int main(void) {\n    return 0;\n}\n",
        "Makefile": "CC ?= gcc\nCFLAGS ?= -Wall\nprog: main.c\n\t$(CC) $(CFLAGS) main.c -o prog\n",
    })
    assert term.ok("make").stdout == "cc -Wall main.c -o prog\n"
    assert term.ok("make -B CFLAGS=-O2").stdout == "cc -O2 main.c -o prog\n"


def test_a_header_typo_in_a_multi_file_build_stops_make(term: Terminal) -> None:
    files = dict(PROJECT_FILES)
    files["survey.c"] = files["survey.c"].replace('"survey.h"', '"surveys.h"')
    term.write(files)
    line = term.run("make")
    assert line.exit_code == 2
    assert "survey.c:2:10: fatal error: surveys.h: No such file or directory" in line.stderr
    assert "make: *** [Makefile:10: survey.o] Error 1" in line.stderr
    assert not term.vfs.exists("/home/user/project/survey_bot")
