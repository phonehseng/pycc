"""Tests for the file and folder commands: ls cd pwd cat mkdir touch rm cp mv chmod find tree file."""
import pytest

from pycc.shell.ansi import DIRECTORY, EXECUTABLE
from tests.shell.conftest import PROJECT, make_program
from pycc.vfs import VirtualFS


@pytest.fixture
def project(vfs: VirtualFS) -> VirtualFS:
    vfs.write_file(f"{PROJECT}/main.c", "int main(void)\n{\n    return 0;\n}\n")
    vfs.write_file(f"{PROJECT}/.secret", "")
    vfs.mkdir(f"{PROJECT}/src")
    vfs.write_file(f"{PROJECT}/src/farm.c", "void till(void) {}\n")
    vfs.write_file(f"{PROJECT}/src/farm.h", "void till(void);\n")
    return vfs


# --- ls ------------------------------------------------------------------------------

def test_ls_into_pipe_is_one_per_line(project, run) -> None:
    assert run.out("ls | cat") == "main.c\nsrc\n"


def test_ls_on_terminal_is_one_line(project, run) -> None:
    assert run.out("ls") == "main.c  src\n"


def test_ls_on_terminal_colours_folders(project, shell) -> None:
    output = shell.run("ls").stdout
    assert f"{DIRECTORY}src" in output and output.count("\n") == 1


def test_ls_colours_executables(vfs, shell) -> None:
    make_program(vfs, f"{PROJECT}/bot")
    assert f"{EXECUTABLE}bot" in shell.run("ls").stdout


def test_ls_all_shows_hidden(project, run) -> None:
    assert run.out("ls -a -1") == ".\n..\n.secret\nmain.c\nsrc\n"


def test_ls_long(project, run) -> None:
    lines = run.out("ls -l").splitlines()
    assert lines[0].startswith("-rw-r--r-- 1 user user")
    assert lines[0].endswith(" main.c")
    assert lines[1].startswith("drwxr-xr-x") and lines[1].endswith(" src")


def test_ls_long_shows_executable_and_read_only(vfs, run) -> None:
    assert run.out("ls -l /usr/include").startswith("-r--r--r--")
    make_program(vfs, f"{PROJECT}/bot")
    assert run.out("ls -l bot").startswith("-rwxr-xr-x")


def test_ls_combined_flags_and_several_folders(project, run) -> None:
    # GNU ls sorts its folder operands too: "." before "src", whatever the typed order.
    assert run.out("ls -1a src .") == (".:\n.\n..\n.secret\nmain.c\nsrc\n\n"
                                       "src:\n.\n..\nfarm.c\nfarm.h\n")
    assert run.out("ls -1 main.c src") == "main.c\n\nsrc:\nfarm.c\nfarm.h\n"


def test_ls_missing(run, shell) -> None:
    result = run("ls nope")
    assert result.exit_code == 2
    assert result.stderr.startswith("ls: cannot access 'nope': No such file or directory")


def test_ls_bad_option(run, shell) -> None:
    result = run("ls -z")
    assert result.exit_code == 2 and "ls: invalid option -- 'z'" in result.stderr
    assert shell.last_diagnostics[0].code == "bad_option"


# --- cd and pwd --------------------------------------------------------------------------

def test_cd_and_pwd(project, run, shell) -> None:
    assert run.out("cd src; pwd") == f"{PROJECT}/src\n"
    assert run.out("cd ..; pwd") == f"{PROJECT}\n"
    assert run.out("cd; pwd") == "/home/user\n"
    assert run.out("cd -") == f"{PROJECT}\n"
    assert shell.env["PWD"] == PROJECT


def test_cd_tilde(run) -> None:
    assert run.out("cd ~/project; pwd") == f"{PROJECT}\n"


def test_cd_errors(project, run, shell) -> None:
    result = run("cd main.c")
    assert result.stderr.startswith("bash: cd: main.c: Not a directory")
    assert shell.last_diagnostics[0].code == "not_a_directory"
    assert run("cd nowhere").stderr.startswith("bash: cd: nowhere: No such file or directory")
    assert "too many arguments" in run("cd a b").stderr


# --- cat, mkdir, touch --------------------------------------------------------------------

def test_cat_files_and_numbers(project, run) -> None:
    assert run.out("cat src/farm.h src/farm.h") == "void till(void);\n" * 2
    assert run.out("cat -n src/farm.h") == "     1\tvoid till(void);\n"


def test_cat_reads_stdin(run) -> None:
    assert run.out("echo piped | cat") == "piped\n"


def test_cat_folder(project, run) -> None:
    result = run("cat src")
    assert result.stderr.startswith("cat: src: Is a directory") and result.exit_code == 1


def test_mkdir(run, vfs) -> None:
    run("mkdir a b")
    assert vfs.is_dir(f"{PROJECT}/a") and vfs.is_dir(f"{PROJECT}/b")
    run("mkdir -p deep/er/est")
    assert vfs.is_dir(f"{PROJECT}/deep/er/est")


def test_mkdir_errors(project, run, shell) -> None:
    assert run("mkdir src").stderr.startswith("mkdir: cannot create directory 'src': File exists")
    assert shell.last_diagnostics[0].code == "file_exists"
    result = run("mkdir x/y")
    assert "No such file or directory" in result.stderr and "mkdir -p" in result.stderr
    assert run("mkdir").stderr.startswith("mkdir: missing operand")
    assert shell.last_diagnostics[0].code == "missing_operand"


def test_touch(vfs, run) -> None:
    run("touch new.txt")
    first = vfs.stat(f"{PROJECT}/new.txt").mtime
    run("touch new.txt")
    assert vfs.read_text(f"{PROJECT}/new.txt") == "" and vfs.stat(f"{PROJECT}/new.txt").mtime > first
    assert "cannot touch 'no/x'" in run("touch no/x").stderr


# --- rm, cp, mv ---------------------------------------------------------------------------

def test_rm_file_and_glob(project, run, vfs) -> None:
    vfs.write_file(f"{PROJECT}/a.o", "")
    vfs.write_file(f"{PROJECT}/b.o", "")
    run("rm *.o main.c")
    assert run.out("ls") == "src\n"


def test_rm_folder_needs_r(project, run, shell, vfs) -> None:
    result = run("rm src")
    assert result.stderr.startswith("rm: cannot remove 'src': Is a directory")
    assert "rm -r src" in result.stderr
    run("rm -r src")
    assert not vfs.exists(f"{PROJECT}/src")


def test_rm_missing_and_force(run) -> None:
    assert run("rm ghost").stderr.startswith("rm: cannot remove 'ghost': No such file or directory")
    assert run("rm -f ghost").exit_code == 0
    assert run("rm -rf ghost").stderr == ""


def test_rm_read_only(run) -> None:
    result = run("rm /usr/include/stdio.h")
    assert "Permission denied" in result.stderr and result.exit_code == 1


def test_rm_refuses_dot(run) -> None:
    assert "refusing to remove" in run("rm -r .").stderr


def test_cp(project, run, vfs) -> None:
    run("cp main.c copy.c")
    assert vfs.read_text(f"{PROJECT}/copy.c") == vfs.read_text(f"{PROJECT}/main.c")
    run("cp main.c src")
    assert vfs.exists(f"{PROJECT}/src/main.c")
    assert run("cp src backup").stderr.startswith("cp: -r not specified; omitting directory 'src'")
    run("cp -r src backup")
    assert vfs.exists(f"{PROJECT}/backup/farm.c")


def test_cp_errors(project, run) -> None:
    assert "missing destination file operand after 'main.c'" in run("cp main.c").stderr
    assert "target 'main.c' is not a directory" in run("cp src/farm.c src/farm.h main.c").stderr
    assert "cannot copy 'nope'" in run("cp nope x").stderr


def test_mv(project, run, vfs) -> None:
    run("mv main.c start.c")
    assert vfs.exists(f"{PROJECT}/start.c") and not vfs.exists(f"{PROJECT}/main.c")
    run("mv start.c src")
    assert vfs.exists(f"{PROJECT}/src/start.c")
    assert "No such file or directory" in run("mv ghost x").stderr
    assert "missing operand" in run("mv").stderr


# --- chmod --------------------------------------------------------------------------------

@pytest.mark.parametrize("mode, expected", [("+x", True), ("u+x", True), ("a+x", True),
                                            ("755", True), ("644", False)])
def test_chmod_modes(vfs, run, mode, expected) -> None:
    vfs.write_file(f"{PROJECT}/run.sh", "echo hi\n")
    assert run(f"chmod {mode} run.sh").exit_code == 0
    assert vfs.stat(f"{PROJECT}/run.sh").executable is expected


def test_chmod_minus_x(vfs, run) -> None:
    vfs.write_file(f"{PROJECT}/run.sh", "")
    run("chmod +x run.sh")
    run("chmod -x run.sh")
    assert not vfs.stat(f"{PROJECT}/run.sh").executable


def test_chmod_errors(vfs, run, shell) -> None:
    assert run("chmod +q f").stderr.startswith("chmod: invalid mode: '+q'")
    assert "missing operand after '+x'" in run("chmod +x").stderr
    assert "cannot access 'ghost'" in run("chmod +x ghost").stderr
    assert "Operation not permitted" in run("chmod +x /usr/include/stdio.h").stderr


# --- find, tree, file ---------------------------------------------------------------------

def test_find_everything(project, run) -> None:
    assert run.out("find") == ".\n./.secret\n./main.c\n./src\n./src/farm.c\n./src/farm.h\n"


def test_find_name_and_type(project, run) -> None:
    assert run.out("find . -name '*.c'") == "./main.c\n./src/farm.c\n"
    assert run.out("find src -type f -name 'farm.*'") == "src/farm.c\nsrc/farm.h\n"
    assert run.out("find . -type d") == ".\n./src\n"


def test_find_errors(run) -> None:
    assert "unknown predicate '-size'" in run("find . -size 3").stderr
    assert "No such file or directory" in run("find nowhere").stderr
    assert "missing argument to '-name'" in run("find . -name").stderr


def test_tree(project, run) -> None:
    assert run.out("tree") == (".\n├── main.c\n└── src\n    ├── farm.c\n    └── farm.h\n"
                               "\n1 directories, 3 files\n")


def test_tree_not_a_folder(project, run) -> None:
    assert run("tree main.c").exit_code == 1


def test_file_kinds(vfs, project, run) -> None:
    make_program(vfs, f"{PROJECT}/bot")
    vfs.write_file(f"{PROJECT}/farm.o", b"CVOBJ1\nxx")
    vfs.write_file(f"{PROJECT}/run.sh", "#!/bin/sh\n")
    output = run.out("file main.c bot farm.o run.sh src nope")
    assert "main.c: C source, ASCII text" in output
    assert "bot: pycc executable" in output
    assert "farm.o: pycc relocatable object" in output
    assert "run.sh: shell script" in output
    assert "src: directory" in output
    assert "nope: cannot open" in output
