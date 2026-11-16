"""Tests for echo and the text filters: head tail wc grep diff sort uniq seq."""
import pytest

from pycc.shell.textcmds import basic_to_python_regex, interpret_escapes
from tests.shell.conftest import PROJECT
from pycc.vfs import VirtualFS


@pytest.fixture
def crops(vfs: VirtualFS) -> VirtualFS:
    vfs.write_file(f"{PROJECT}/crops.txt", "corn\nWheat\ncorn\nbeet\ncorn\nwheat\n")
    vfs.write_file(f"{PROJECT}/nums.txt", "10\n9\n100\n-1\n")
    vfs.write_file(f"{PROJECT}/twelve.txt", "".join(f"line {i}\n" for i in range(1, 13)))
    return vfs


# --- echo ------------------------------------------------------------------------------

@pytest.mark.parametrize("line, output", [
    ("echo a b", "a b\n"),
    ("echo -n a", "a"),
    ("echo -e 'a\\tb\\nc'", "a\tb\nc\n"),
    ("echo 'a\\nb'", "a\\nb\n"),
    ("echo -ne 'x\\n'", "x\n"),
    ("echo -e 'stop\\cnever'", "stop"),
    ("echo -x", "-x\n"),
    ("echo", "\n"),
])
def test_echo(run, line, output) -> None:
    assert run.out(line) == output


def test_echo_escape_codes_for_colour() -> None:
    text, stopped = interpret_escapes("\\e[32mgreen\\033[0m \\x41")
    assert text == "\x1b[32mgreen\x1b[0m A" and not stopped


# --- head and tail ------------------------------------------------------------------------

def test_head_default_ten(crops, run) -> None:
    assert run.out("head twelve.txt").count("\n") == 10


@pytest.mark.parametrize("line, output", [
    ("head -n 2 crops.txt", "corn\nWheat\n"),
    ("head -n2 crops.txt", "corn\nWheat\n"),
    ("head -2 crops.txt", "corn\nWheat\n"),
    ("tail -n 2 crops.txt", "corn\nwheat\n"),
    ("tail -1 crops.txt", "wheat\n"),
    ("tail -n 0 crops.txt", ""),
    ("seq 5 | head -n 1", "1\n"),
    ("seq 5 | tail -n 1", "5\n"),
])
def test_head_tail(crops, run, line, output) -> None:
    assert run.out(line) == output


def test_head_several_files_have_headers(crops, run) -> None:
    assert run.out("head -n 1 crops.txt nums.txt") == "==> crops.txt <==\ncorn\n\n==> nums.txt <==\n10\n"


def test_head_bad_count(run, shell) -> None:
    assert "invalid number of lines: 'x'" in run("head -n x f").stderr
    assert "option requires an argument" in run("head -n").stderr


# --- wc -----------------------------------------------------------------------------------

@pytest.mark.parametrize("line, output", [
    ("wc -l crops.txt", "6 crops.txt\n"),
    ("wc -w crops.txt", "6 crops.txt\n"),
    ("wc -c crops.txt", "32 crops.txt\n"),
    ("cat crops.txt | wc -l", "6\n"),
    ("wc -l < crops.txt", "6\n"),
    ("echo 'one two' | wc", "      1       2       8\n"),
])
def test_wc(crops, run, line, output) -> None:
    assert run.out(line) == output


def test_wc_total(crops, run) -> None:
    assert run.out("wc -l crops.txt nums.txt") == " 6 crops.txt\n 4 nums.txt\n10 total\n"


# --- grep ---------------------------------------------------------------------------------

@pytest.mark.parametrize("line, output", [
    ("grep corn crops.txt", "corn\ncorn\ncorn\n"),
    ("grep -n corn crops.txt", "1:corn\n3:corn\n5:corn\n"),
    ("grep -c corn crops.txt", "3\n"),
    ("grep -i wheat crops.txt", "Wheat\nwheat\n"),
    ("grep -v corn crops.txt", "Wheat\nbeet\nwheat\n"),
    ("grep -vc corn crops.txt", "3\n"),
    ("grep -in WHEAT crops.txt", "2:Wheat\n6:wheat\n"),
    ("grep '^b' crops.txt", "beet\n"),
    ("grep -E 'beet|Wheat' crops.txt", "Wheat\nbeet\n"),
    ("grep 'beet|Wheat' crops.txt", ""),
    ("grep 'beet\\|Wheat' crops.txt", "Wheat\nbeet\n"),
    ("grep -E 'e+t' crops.txt", "beet\n"),
    ("grep -E 'ea?t' crops.txt", "Wheat\nbeet\nwheat\n"),
    ("grep 'ea\\?t' crops.txt", "Wheat\nbeet\nwheat\n"),
])
def test_grep(crops, run, line, output) -> None:
    assert run.out(line) == output


def test_grep_exit_codes(crops, run) -> None:
    assert run("grep corn crops.txt").exit_code == 0
    assert run("grep rice crops.txt").exit_code == 1
    assert run("grep rice nope.txt").exit_code == 2


def test_grep_several_files_prefix_names(crops, run) -> None:
    assert run.out("grep -c 1 crops.txt nums.txt") == "crops.txt:0\nnums.txt:3\n"


def test_grep_recursive(vfs, run) -> None:
    vfs.mkdir(f"{PROJECT}/src")
    vfs.write_file(f"{PROJECT}/src/a.c", "int main(void);\n")
    vfs.write_file(f"{PROJECT}/b.c", "int main(void) {}\n")
    assert run.out("grep -r main") == "b.c:int main(void) {}\nsrc/a.c:int main(void);\n"
    assert run.out("grep -rn main src") == "src/a.c:1:int main(void);\n"


def test_grep_folder_without_r(vfs, run) -> None:
    vfs.mkdir(f"{PROJECT}/src")
    assert "grep: src: Is a directory" in run("grep x src").stderr


def test_grep_from_pipe(run) -> None:
    assert run.out("seq 12 | grep 1") == "1\n10\n11\n12\n"


def test_grep_missing_pattern_and_bad_regex(run, shell) -> None:
    assert run("grep").exit_code == 2
    result = run("grep -E '(' f")
    assert result.exit_code == 2 and "invalid regular expression" in result.stderr
    assert shell.last_diagnostics[0].code == "bad_regex"


@pytest.mark.parametrize("basic, python", [
    ("a+b", "a\\+b"), ("a\\+b", "a+b"), ("(x)", "\\(x\\)"), ("\\(x\\)", "(x)"),
    ("a|b", "a\\|b"), ("a\\|b", "a|b"), ("x{2}", "x\\{2\\}"), ("a.*\\.c", "a.*\\.c"),
])
def test_basic_regex_translation(basic, python) -> None:
    assert basic_to_python_regex(basic) == python


# --- diff ---------------------------------------------------------------------------------

@pytest.fixture
def versions(vfs: VirtualFS) -> None:
    vfs.write_file(f"{PROJECT}/a.txt", "one\ntwo\nthree\n")
    vfs.write_file(f"{PROJECT}/b.txt", "one\n2\nthree\nfour\n")
    vfs.write_file(f"{PROJECT}/same.txt", "one\ntwo\nthree\n")


def test_diff_same_files(versions, run) -> None:
    result = run("diff a.txt same.txt")
    assert result.stdout == "" and result.exit_code == 0


def test_diff_normal(versions, run) -> None:
    result = run("diff a.txt b.txt")
    assert result.stdout == "2c2\n< two\n---\n> 2\n3a4\n> four\n" and result.exit_code == 1


def test_diff_unified(versions, run) -> None:
    lines = run.out("diff -u a.txt b.txt").splitlines()
    assert lines[:3] == ["--- a.txt", "+++ b.txt", "@@ -1,3 +1,4 @@"]
    assert "-two" in lines and "+2" in lines and "+four" in lines


def test_diff_errors(versions, run) -> None:
    assert run("diff a.txt").exit_code == 2
    assert run("diff a.txt nope").exit_code == 2


# --- sort, uniq, seq ----------------------------------------------------------------------

@pytest.mark.parametrize("line, output", [
    ("sort crops.txt", "Wheat\nbeet\ncorn\ncorn\ncorn\nwheat\n"),
    ("sort -r crops.txt", "wheat\ncorn\ncorn\ncorn\nbeet\nWheat\n"),
    ("sort nums.txt", "-1\n10\n100\n9\n"),
    ("sort -n nums.txt", "-1\n9\n10\n100\n"),
    ("sort -rn nums.txt", "100\n10\n9\n-1\n"),
])
def test_sort(crops, run, line, output) -> None:
    assert run.out(line) == output


def test_uniq(crops, run) -> None:
    assert run.out("sort crops.txt | uniq") == "Wheat\nbeet\ncorn\nwheat\n"
    assert run.out("sort crops.txt | uniq -c") == "      1 Wheat\n      1 beet\n      3 corn\n      1 wheat\n"
    assert run.out("uniq crops.txt") == "corn\nWheat\ncorn\nbeet\ncorn\nwheat\n"


@pytest.mark.parametrize("line, output", [
    ("seq 3", "1\n2\n3\n"),
    ("seq 2 4", "2\n3\n4\n"),
    ("seq 0 5 10", "0\n5\n10\n"),
    ("seq 3 -1 1", "3\n2\n1\n"),
    ("seq 0.5 0.5 1.5", "0.5\n1\n1.5\n"),
    ("seq 5 1", ""),
])
def test_seq(run, line, output) -> None:
    assert run.out(line) == output


def test_seq_errors(run) -> None:
    assert "invalid floating point argument: 'x'" in run("seq x").stderr
    assert "missing operand" in run("seq").stderr
    assert "Zero increment" in run("seq 1 0 5").stderr
