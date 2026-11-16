"""Review tests for the built-in commands: cd, rm, cp, mv, grep, seq and real-disk safety."""
import re
from pathlib import Path

from pycc.shell.textcmds import SEQ_LIMIT
from tests.shell.conftest import PROJECT, make_program

SHELL_DIR = Path(__file__).resolve().parents[2] / "pycc" / "shell"


# --- cd -------------------------------------------------------------------------------------

def test_cd_without_arguments_goes_home(run, shell) -> None:
    assert run("cd").exit_code == 0
    assert shell.cwd == "/home/user"


def test_cd_dash_before_any_cd_reports_oldpwd_not_set(run, shell) -> None:
    result = run("cd -")
    assert result.exit_code == 1
    assert "OLDPWD not set" in result.stderr
    assert shell.cwd == PROJECT
    assert result.finished.diagnostics[0].friendly


def test_cd_dash_goes_back_and_prints_the_folder(run, shell) -> None:
    run("cd /tmp")
    assert run.out("cd -") == PROJECT + "\n"
    assert shell.cwd == PROJECT


def test_cd_normalises_dot_dot(run, shell) -> None:
    run("mkdir -p src/lib && cd src/lib/../../src/./lib/..")
    assert shell.cwd == f"{PROJECT}/src"
    run("cd ../../../../../..")
    assert shell.cwd == "/"


# --- rm ---------------------------------------------------------------------------------------

def test_rm_recursive_on_root_is_refused(run, vfs) -> None:
    for line in ("rm -r /", "rm -rf /", "rm -rf /home/..", "cd / && rm -rf ."):
        result = run(line)
        assert result.exit_code == 1, line
    assert vfs.exists(PROJECT) and vfs.exists("/tmp")
    assert "dangerous" in run("cd / ; rm -r /").stderr


def test_rm_root_diagnostic_has_code_and_friendly(run) -> None:
    diagnostic = run("rm -rf /").finished.diagnostics[0]
    assert diagnostic.code == "permission_denied"
    assert diagnostic.friendly and diagnostic.message.startswith("rm:")


# --- cp and mv ---------------------------------------------------------------------------------

def test_cp_file_onto_itself_reports_same_file(run, vfs) -> None:
    vfs.write_file(f"{PROJECT}/a.txt", "data")
    result = run("cp a.txt a.txt")
    assert result.exit_code == 1
    assert "cp: 'a.txt' and 'a.txt' are the same file" in result.stderr
    assert result.finished.diagnostics[0].code == "same_file"
    assert vfs.read_text(f"{PROJECT}/a.txt") == "data"


def test_cp_r_directory_onto_existing_directory(run, vfs) -> None:
    run("mkdir -p src backup && echo x > src/main.c")
    assert run("cp -r src backup").exit_code == 0
    assert vfs.read_text(f"{PROJECT}/backup/src/main.c") == "x\n"


def test_mv_rename_across_directories(run, vfs) -> None:
    run("mkdir a b && echo hi > a/old.txt")
    assert run("mv a/old.txt b/new.txt").exit_code == 0
    assert not vfs.exists(f"{PROJECT}/a/old.txt")
    assert vfs.read_text(f"{PROJECT}/b/new.txt") == "hi\n"


def test_mv_to_missing_folder_with_slash_does_not_rename(run, vfs) -> None:
    vfs.write_file(f"{PROJECT}/notes.txt", "n")
    result = run("mv notes.txt docs/")
    assert result.exit_code == 1
    assert "mv: cannot move 'notes.txt' to 'docs/': Not a directory" in result.stderr
    assert vfs.exists(f"{PROJECT}/notes.txt") and not vfs.exists(f"{PROJECT}/docs")


def test_cp_to_missing_folder_with_slash_fails(run, vfs) -> None:
    vfs.write_file(f"{PROJECT}/notes.txt", "n")
    assert run("cp notes.txt docs/").exit_code == 1
    assert not vfs.exists(f"{PROJECT}/docs")


# --- grep -------------------------------------------------------------------------------------

def test_grep_binary_file_reports_a_match_instead_of_garbage(run, vfs) -> None:
    vfs.write_file(f"{PROJECT}/data.bin", bytes([0, 1]) + b"corn" + bytes([255]) + b"\n")
    result = run("grep corn data.bin")
    assert result.exit_code == 0
    assert result.stdout == "grep: data.bin: binary file matches\n"


def test_grep_compiled_program_is_binary(run, vfs) -> None:
    make_program(vfs, f"{PROJECT}/bot", "corn")
    assert run.out("grep corn bot") == "grep: bot: binary file matches\n"
    assert run("grep wheat bot").exit_code == 1


def test_grep_binary_count_still_counts(run, vfs) -> None:
    vfs.write_file(f"{PROJECT}/data.bin", bytes([0]) + b"a\na\nb\n")
    assert run.out("grep -c a data.bin") == "2\n"


def test_grep_error_wins_over_a_match(run, vfs) -> None:
    vfs.write_file(f"{PROJECT}/a.txt", "corn\n")
    result = run("grep corn a.txt missing.txt")
    assert "a.txt:corn" in result.stdout
    assert result.exit_code == 2


# --- seq --------------------------------------------------------------------------------------

def test_seq_float_steps_reach_the_end_exactly(run) -> None:
    assert run.out("seq 0 0.1 0.5") == "0\n0.1\n0.2\n0.3\n0.4\n0.5\n"


def test_seq_wrong_direction_prints_nothing(run) -> None:
    result = run("seq 5 1")
    assert result.exit_code == 0 and result.stdout == ""


def test_seq_refuses_huge_ranges(run) -> None:
    result = run(f"seq {SEQ_LIMIT * 10}")
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.finished.diagnostics[0].friendly


# --- scripts ----------------------------------------------------------------------------------

def test_sh_on_a_folder_explains_it_is_a_folder(run) -> None:
    run("mkdir tools")
    result = run("sh tools")
    assert result.exit_code == 126
    assert "folder" in result.finished.diagnostics[0].friendly


# --- the shell never touches the real disk ----------------------------------------------------

def test_shell_source_never_opens_real_files_or_uses_os() -> None:
    forbidden = re.compile(r"\bopen\(|\bimport os\b|\bos\.|\bshutil\b|\bsubprocess\b|"
                           r"\bsocket\b|\burllib\b|\bPath\(")
    offenders = []
    for path in sorted(SHELL_DIR.glob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if forbidden.search(line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert offenders == []
