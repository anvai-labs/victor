"""Vendored SCIP protobuf bindings (private; generated).

Generated with protoc from scip.proto as shipped by scip-code/scip
(Apache-2.0; the successor home of Sourcegraph's SCIP schema — the go
package is ``org.scip_code.scip``). See scip.proto alongside for the exact
source; this schema introduced the flat ``single_line_range`` /
``multi_line_range`` typed ranges that replaced the nested Position shape of
the older Sourcegraph proto. The official scip-python toolchain is npm-only
(no pip package provides these bindings, and PyPI's ``scip`` is an unrelated
project), so the schema is vendored here rather than imported.
"""

from . import scip_pb2

__all__ = ["scip_pb2"]
