"""
Convert a Python script containing triple-quoted Markdown strings to a Markdown document.

See the README for details on the conversion process.

# TBD: validate HTML, especially details and summary tags that don't match

CM: removed unidecode because it wasn't working correctly in some cases. 
Run fix_special_characters.py on the solutions BEFORE running this instead
"""

import ast
import collections
import os
import re
import stat
import string
from dataclasses import dataclass
from typing import Any

import black
import click


def strip_trailing_spaces(s: str) -> str:
    return re.sub(r"[ \t]+\n", "\n", s)


class StripSolutions(ast.NodeTransformer):
    def visit_If(self, node):
        """Strip out contents of if 'SOLUTION': block."""
        if type(node.test) is ast.Constant and node.test.value == "SOLUTION":
            if node.orelse:
                return node.orelse
            return ast.Expr(ast.Constant(value="TODO: YOUR CODE HERE"))
        if type(node.test) is ast.Constant and node.test.value == "SKIP":
            return None
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Strip out remainder of function body after 'SOLUTION' constant."""
        self.generic_visit(node)  # transform the body
        for i, child in enumerate(node.body):
            if type(child) is ast.Expr and type(child.value) is ast.Constant and child.value.value == "SOLUTION":
                # Let's always insert a pass statement to ensure it's a legal block still
                # For example if it was just docstring and then "SOLUTION".
                node.body = node.body[:i] + [ast.Expr(ast.Constant(value="TODO: YOUR CODE HERE")), ast.Pass()]
                break
        return node


@dataclass
class Snippet:
    language: str
    text: str


def is_toplevel_string_constant(node):
    """Return True if this is a string constant that's unindented."""
    if not (type(node) is ast.Expr and type(node.value) is ast.Constant):
        return False

    if type(node.value.value) is not str:
        return False

    # Check if it's at toplevel - node.value doesn't have col_offset in some versions of Python
    if getattr(node, "col_offset", None) == 0:
        return True

    if getattr(node.value, "col_offset", None) == 0:
        return True

    return False


TOC_LEVELS = [
    "-",
    "    -",
    "        -",
]

TOC_RE = re.compile(r"^(#+) (.+)$")
TOC_MARKER = "<!-- toc -->"
SLUG_ALLOWED_CHARS = string.ascii_letters + "-"


@dataclass
class TOCEntry:
    title: str
    level: int  # zero-indexed, e.g. # is level 0
    slug: str  # spaces to dashes, possible -dash-number


TAG_RE = re.compile(r"<(/?)([a-zA-Z]+?)(/?)>")
NO_CLOSE_TAGS = ["br"]


def check_html_tags(text: str) -> list[str]:
    """Return a list of warnings about mismatched HTML tags."""
    tags = TAG_RE.findall(text)
    tagname_stack: list[str] = []
    warnings: list[str] = []
    for tag in tags:
        close_slash, tagname, end_slash = tag
        if close_slash:
            if end_slash:
                warnings.append(f"WARNING: malformed HTML tag {tag} ")
            else:
                if tagname_stack:
                    top = tagname_stack.pop()
                    if top == tagname:
                        continue  # proper close
                    else:
                        warnings.append(f"WARNING: should have closed {top} but found {tag}")
                else:
                    warnings.append(f"WARNING: tried to close {tag} but no tags open")
        else:
            if end_slash:
                continue  # is self-closing
            elif tagname in NO_CLOSE_TAGS:
                continue
            else:
                tagname_stack.append(tagname)
    if tagname_stack:
        warnings.append(f"WARNING: tags not closed: {tagname_stack}")
    return warnings


class SolutionMaker(ast.NodeVisitor):
    def __init__(self):
        self.snippets = []
        self.mode = black.Mode(line_length=120)  # type: ignore
        self.toc_entries: list[TOCEntry] = []
        self.counters = collections.defaultdict(int)

    def generic_visit(self, node):
        if type(node) is ast.Module:
            super().generic_visit(node)

        elif is_toplevel_string_constant(node):
            text = strip_trailing_spaces(node.value.value)  # type: ignore
            text = self._maybe_add_toc(text)
            warnings = check_html_tags(text)
            if warnings:
                print(f"Bad HTML tags between lines {node.lineno} and {node.end_lineno}")
                print("\n".join(warnings))

            self.snippets.append(Snippet("markdown", text))  # type: ignore

        else:
            # Concat with previous code block if possible
            src = ast.unparse(node)
            if self.snippets and self.snippets[-1].language == "python":
                self.snippets[-1].text += "\n" + src
            else:
                self.snippets.append(Snippet("python", src))

    def dump(self, fp):
        texts = []
        last_language = None
        for snippet in self.snippets:
            if snippet.language == "markdown":
                text = snippet.text
                index = text.find(TOC_MARKER)
                if index != -1:
                    toc = self._dump_toc()
                    text = text[:index] + toc + text[index + len(TOC_MARKER) :]
                texts.append(text)
            elif snippet.language == "python":
                pretty = black.format_str(snippet.text, mode=self.mode)
                # Need two newlines before code block
                extra_newline = "\n" if last_language == "markdown" else ""
                texts.append(f"{extra_newline}```python\n{pretty}\n```")
            last_language = snippet.language
        fp.write("\n".join(texts))

    def _maybe_add_toc(self, text: str) -> str:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            m = TOC_RE.match(line)
            if m is not None:
                pounds, header_text = m.groups()
                prefix = header_text.replace(" ", "-")
                count = self.counters[prefix]
                slug = f"{prefix}-{count}" if count > 0 else prefix
                slug = slug.lower()  # VSCode only wants lowercase slugs
                slug = "".join(c for c in slug if c in SLUG_ALLOWED_CHARS)
                level = len(pounds) - 2
                if level < 0:
                    continue  # Don't need to repeat toplevel
                if level >= len(TOC_LEVELS):
                    raise ValueError(f"TOC doesn't yet support header level {level}: {header_text}")
                entry = TOCEntry(title=header_text, level=level, slug=slug)
                self.toc_entries.append(entry)
                self.counters[prefix] = count + 1
                # lines[i] = f'{pounds} <a id="#{slug}">{header_text}</a>'
        return "\n".join(lines) + "\n"  # trailing newline is needed for MD031

    def _dump_toc(self) -> str:
        lines = ["## Table of Contents", ""]
        for e in self.toc_entries:
            line = f"{TOC_LEVELS[e.level]} [{e.title}](#{e.slug})"
            lines.append(line)
        return "\n".join(lines)


def guess_outname(input_name):
    assert input_name.endswith("_solution.py")
    outname = input_name.replace("_solution.py", "_instructions.md")
    return outname


def build(input, output):
    print(f"Building: {input.name} -> {output.name}")
    s = input.read()
    m = ast.parse(s)
    StripSolutions().visit(m)
    sm = SolutionMaker()
    sm.visit(m)
    sm.dump(output)


@click.command()
@click.argument("input", type=click.File("rb"))
@click.argument("output", type=click.File("w"), required=False)
def build_cmd(input, output):
    if output:
        result = build(input, output)
    else:
        outname = guess_outname(input.name)
        if os.path.exists(outname):
            os.chmod(outname, stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH)
        with open(outname, "w") as f:
            result = build(input, f)
        os.chmod(outname, 0o777 ^ (stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH))
    return result


if __name__ == "__main__":
    build_cmd()
