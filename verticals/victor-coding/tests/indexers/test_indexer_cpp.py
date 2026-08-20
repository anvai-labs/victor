# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

pytest.importorskip("victor_coding.codebase.indexer")

from pathlib import Path
from victor_coding.codebase.indexer import CodebaseIndex


@pytest.mark.skip(reason="Tree-sitter C++ parser not available in CI environment")
@pytest.mark.asyncio
async def test_cpp_indexing(tmp_path: Path):
    """Test indexing of a simple C++ file."""
    cpp_content = """
    #include <iostream>

    class MyClass {
    public:
        void myMethod() {
            std::cout << "Hello, World!" << std::endl;
        }
    };

    int main() {
        MyClass obj;
        obj.myMethod();
        return 0;
    }
    """
    cpp_file = tmp_path / "main.cpp"
    cpp_file.write_text(cpp_content)

    indexer = CodebaseIndex(str(tmp_path))
    await indexer.index_codebase()

    assert "main.cpp" in indexer.files
    file_metadata = indexer.files["main.cpp"]

    symbols = {s.name: s for s in file_metadata.symbols}
    assert "MyClass" in symbols
    assert "myMethod" in symbols
    assert "main" in symbols

    assert symbols["MyClass"].type == "class"
    assert symbols["myMethod"].type == "function"
    assert symbols["main"].type == "function"

    call_edges = file_metadata.call_edges
    assert ("main", "myMethod") in call_edges
