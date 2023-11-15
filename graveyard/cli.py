import cmd
import os
import textwrap


class DataShell(cmd.Cmd):
    intro = "Test intro"
    prompt = "\033[214mData >\033[0m"

    def emptyline(self):
        return

    def do_foo(self, args):
        print("Foo")

    def do_bar(self, args):
        print("Bar")


if __name__ == '__main__':
    DataShell().cmdloop()
