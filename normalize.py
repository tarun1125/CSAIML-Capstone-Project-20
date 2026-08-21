import ast
import json
import re
import sys

# Defaults keep this a no-argument, Qwen-only script like before. Pass two
# paths on the command line to reuse it for the GPT arm instead of
# duplicating the whole file:
#   python normalize.py data/gpt_results.json data/gpt_normalized.json
INPUT = sys.argv[1] if len(sys.argv) > 1 else "data/claude_baseline_results.json"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "data/claude_normalized.json"

# Qwen was trained mostly on MongoDB-shell/JS-flavored docs, not PyMongo's
# Python API, and it shows up as two confirmed, recurring dialect habits
# that evaluation/execute_queries.py's safety check correctly rejects (it's
# not wrong to reject them -- `null` isn't a Python name and `.countDocuments`
# isn't a real PyMongo method):
#   - bare JSON `null`/`true`/`false` where Python needs `None`/`True`/`False`
#     -- confirmed via 5 RAG-arm + 10 baseline Qwen "REJECTED by safety check
#     (unexpected name: null)" errors, 0 on Claude.
#   - camelCase Mongo-shell method names (`countDocuments`, `findOne`)
#     instead of PyMongo's snake_case (`count_documents`, `find_one`) --
#     confirmed via 2 RAG-arm "method not allowed: countDocuments" errors.
# Fixed here, once, at the AST level rather than as a regex/string
# replacement: `null` as a bare identifier is an ast.Name node, while the
# literal string "null" in quotes (a real field value some of this data
# actually has) is an ast.Constant node -- so this only ever touches the
# unquoted token, never a quoted string that happens to spell the same word.
_LITERAL_REWRITES = {"null": None, "true": True, "false": False}
_METHOD_ALIASES = {"countDocuments": "count_documents", "findOne": "find_one"}


class _DialectFixer(ast.NodeTransformer):
    """Rewrites the two confirmed MongoDB-shell/JS habits above into valid
    Python/PyMongo. Records what it changed (for the per-id log line below)
    -- traceability matters here since this is a silent-by-default rewrite
    of the model's actual output."""

    def __init__(self):
        self.notes = []

    def visit_Name(self, node):
        if node.id in _LITERAL_REWRITES:
            value = _LITERAL_REWRITES[node.id]
            self.notes.append(f"bare `{node.id}` -> `{value!r}`")
            return ast.copy_location(ast.Constant(value=value), node)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)  # fix any dialect issue nested inside first
        if node.attr in _METHOD_ALIASES:
            self.notes.append(f".{node.attr}() -> .{_METHOD_ALIASES[node.attr]}()")
            node.attr = _METHOD_ALIASES[node.attr]
        return node

    def visit_Call(self, node):
        self.generic_visit(node)  # renames node.func.attr first, if applicable
        # PyMongo's count_documents() requires an explicit filter argument --
        # unlike the Mongo-shell countDocuments() it's often rewritten from
        # above, which defaults to counting everything when called with zero
        # arguments. Confirmed via a real post-rename crash ("count_documents()
        # missing 1 required positional argument: 'filter'", case cs-e2):
        # renaming the method name alone is necessary but not sufficient for
        # this zero-arg case. find_one() is NOT included here -- its filter
        # argument is genuinely optional in PyMongo (defaults to None), so a
        # zero-arg .find_one() is already valid and needs no rewrite.
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "count_documents"
            and not node.args
            and not node.keywords
        ):
            self.notes.append("count_documents() with no args -> count_documents({})")
            node.args = [ast.Dict(keys=[], values=[])]
        return node


def normalize(query: str) -> str:
    query = query.strip()

    try:
        tree = ast.parse(query, mode="eval")
    except SyntaxError:
        # Leave unparseable input alone. execute_queries.py will correctly
        # report this as a parse failure -- better than silently mangling it
        # into something that merely looks plausible.
        print(f"[normalize] WARNING could not parse, leaving as-is: {query[:80]}")
        return re.sub(r"\s+", " ", query).strip()

    node = tree.body
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
        and len(node.args) == 1
    ):
        node = node.args[0]

    fixer = _DialectFixer()
    node = fixer.visit(node)
    ast.fix_missing_locations(node)
    if fixer.notes:
        print(f"[normalize] dialect fix applied: {'; '.join(fixer.notes)}")

    return ast.unparse(node)


if __name__ == "__main__":
    with open(INPUT, encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["normalized_query"] = normalize(item["generated_query"])
        print(f"[normalize] id={item.get('id')}: {item['normalized_query'][:80]}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[normalize] wrote {len(data)} rows -> {OUTPUT}")
