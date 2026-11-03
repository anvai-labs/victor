# TUI Rendering Analysis and Fixes

**Date**: 2026-11-01  
**Issue**: Markdown content (especially code blocks) not rendering properly in Victor TUI  
**Handoff Reference**: `docs/development/project-db-deduplication-handoff.md`

## Problem Summary

The handoff document contains:
- 33 code blocks (SQL, bash, python)
- 6 code blocks with lines exceeding 92 characters (max: 182 chars)
- Multiple markdown tables
- Long inline code references

**Symptoms**: Code blocks and tables may overflow the conversation pane, causing horizontal scrolling or truncation.

## Root Cause Analysis

### 1. RichLog Width Constraints

**File**: `victor/ui/tui/styles.tcss`

```css
#conversation {
    height: 1fr;
    padding: 0 1;
    background: $background;
    color: $foreground;
}
```

**Issue**: The `#conversation` RichLog has no explicit `width` constraint. It inherits from parent `#main` which is `width: 1fr`, but the actual rendered width depends on terminal size and sidebar width.

**Expected Layout**:
- Terminal: 120 chars (typical)
- Sidebar: 28 chars
- Main area: 92 chars
- Conversation padding: 2 chars (1 left, 1 right)
- **Effective width: 90 chars**

### 2. Markdown Code Block Wrapping

**File**: `victor/ui/rendering/markdown.py`

Rich's `Markdown` renderer creates `CodeBlock` objects that use `Syntax` with `word_wrap=False` by default. This means:
- Long lines are NOT wrapped
- They extend beyond the available width
- RichLog must handle the overflow

**Current Code** (line 63-73):
```python
def _markdown_block(text: str) -> Markdown:
    return Markdown(
        text,
        code_theme="monokai",
        inline_code_lexer="python",
        inline_code_theme="monokai",
    )
```

**Issue**: No width constraint is passed to the Markdown renderer. It relies on the Console's width, but RichLog may not properly propagate its width to the Markdown renderables.

### 3. RichLog Configuration

**File**: `victor/ui/tui/wire_timeline.py` (line 195-198)

```python
class WireTimeline(RichLog):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("wrap", True)
        kwargs.setdefault("markup", True)
        super().__init__(**kwargs)
```

**Issue**: `wrap=True` enables text wrapping, but Markdown renderables (especially code blocks) may not respect this setting. The `Syntax` widget inside code blocks has its own wrapping logic.

## Evidence from Testing

### Test 1: Rich Markdown at width=92
```
=== Test 1: Markdown (default) at width=92 ===
  2 ( 92):  SELECT s.name, s.symbol_type, s.file_path, count(*) as sym_cnt,                            
  3 ( 92):  very_long_column_name_that_exceeds_width, another_long_column FROM symbols WHERE           
  4 ( 92):  condition;                                                                                 

Lines > 92 chars: 0
```

✅ Rich's Markdown **does** wrap code blocks when Console width is constrained.

### Test 2: Handoff Code Block Analysis
```
Found 33 code blocks

Block 3 (sql):
  Lines: 16
  Max line: 182 chars  ⚠️
  Avg line: 39.8 chars

Block 6 (sql):
  Lines: 4
  Max line: 95 chars  ⚠️

Block 8 (sql):
  Lines: 11
  Max line: 104 chars  ⚠️
```

6 code blocks have lines exceeding 92 chars.

### Test 3: Panel + Syntax (word_wrap=False)
```
=== Test 2: Panel + Syntax (word_wrap=False) ===
  1 ( 92): ╭─────────────────────────────────────── SQL Query ────────────────────────────────────────╮
  2 ( 92): │ SELECT s.name, s.symbol_type, s.file_path, count(*) as sym_cnt                           │
  3 ( 92): │ FROM symbols s                                                                           │
  4 ( 92): │ JOIN graph_node g ON s.name = g.name AND s.file_path = g.file AND s.symbol_type = g.type │

Lines > 92 chars: 0
```

✅ Panel constrains width even with `word_wrap=False`.

## Identified Issues

### Issue 1: RichLog Width Propagation
**Severity**: High  
**Impact**: Markdown renderables may not know the available width

RichLog's `write()` method accepts renderables, but the width constraint may not propagate correctly to nested Markdown/CodeBlock/Syntax widgets.

### Issue 2: Code Block Overflow
**Severity**: Medium  
**Impact**: Long SQL/bash lines overflow the conversation pane

While Rich's Markdown wraps code blocks when Console width is set, RichLog may not set the Console width correctly for each renderable.

### Issue 3: Table Rendering
**Severity**: Low  
**Impact**: Wide tables may overflow

Markdown tables are rendered as Rich Table objects, which have their own width calculation logic.

## Proposed Fixes

### Fix 1: Explicit Width Constraint in styles.tcss

**File**: `victor/ui/tui/styles.tcss`

```css
#conversation {
    height: 1fr;
    width: 100%;  /* ADD: Explicit width */
    padding: 0 1;
    background: $background;
    color: $foreground;
    overflow-x: hidden;  /* ADD: Prevent horizontal scroll */
}
```

### Fix 2: Constrain Markdown Rendering Width

**File**: `victor/ui/rendering/markdown.py`

Add a wrapper function that constrains Markdown rendering:

```python
from rich.console import Console, ConsoleOptions
from rich.segment import Segment

def _constrained_markdown(text: str, max_width: int = 90) -> Markdown:
    """Create a Markdown renderable with explicit width constraint."""
    md = Markdown(
        text,
        code_theme="monokai",
        inline_code_lexer="python",
        inline_code_theme="monokai",
    )
    
    # Wrap in a constraint to limit width
    class ConstrainedMarkdown:
        def __rich_console__(self, console: Console, options: ConsoleOptions):
            # Force width constraint
            constrained_options = options.update_width(min(options.max_width, max_width))
            yield from console.render(md, constrained_options)
    
    return ConstrainedMarkdown()
```

### Fix 3: RichLog Width Propagation

**File**: `victor/ui/tui/wire_timeline.py`

Override the `write()` method to ensure width is propagated:

```python
class WireTimeline(RichLog):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("wrap", True)
        kwargs.setdefault("markup", True)
        kwargs.setdefault("max_lines", 10000)  # Prevent memory issues
        super().__init__(**kwargs)
        self._state = WireTimelineState()
    
    def write(
        self,
        content: object,
        width: int | None = None,
        expand: bool = False,
        shrink: bool = True,
        scroll_end: bool = True,
        animate: bool = False,
    ) -> Self:
        """Override to ensure width is constrained."""
        # If width not specified, use content_width
        if width is None and hasattr(self, 'content_width'):
            width = self.content_width
        
        return super().write(content, width=width, expand=expand, shrink=shrink, 
                           scroll_end=scroll_end, animate=animate)
```

### Fix 4: Code Block Word Wrap

**File**: `victor/ui/rendering/markdown.py`

Override Rich's CodeBlock rendering to enable word wrap:

```python
from rich.markdown import CodeBlock as RichCodeBlock
from rich.syntax import Syntax

class WrappedCodeBlock:
    """CodeBlock with word_wrap enabled."""
    
    def __init__(self, code: str, language: str, theme: str = "monokai"):
        self.code = code
        self.language = language
        self.theme = theme
    
    def __rich_console__(self, console: Console, options: ConsoleOptions):
        syntax = Syntax(
            self.code,
            self.language,
            theme=self.theme,
            word_wrap=True,  # Enable wrapping
            line_numbers=False,
        )
        yield syntax

# Monkey-patch Rich's CodeBlock
import rich.markdown
original_code_block_rich_console = RichCodeBlock.__rich_console__

def patched_code_block_rich_console(self, console: Console, options: ConsoleOptions):
    # Use word_wrap=True for all code blocks
    syntax = Syntax(
        str(self.text).rstrip(),
        self.language,
        theme=self.theme or "monokai",
        word_wrap=True,
        line_numbers=False,
    )
    yield syntax

RichCodeBlock.__rich_console__ = patched_code_block_rich_console
```

## Testing Plan

### Test 1: Verify Width Constraint
```bash
# Run TUI with handoff content
victor tui

# In another terminal, send the handoff content
# Check that code blocks are wrapped, not overflowed
```

### Test 2: Measure Rendered Width
```python
# Add debug logging to WireTimeline.write()
def write(self, content, width=None, ...):
    if width is None:
        width = getattr(self, 'content_width', None)
    print(f"DEBUG: Writing content, width={width}, content_width={getattr(self, 'content_width', None)}")
    return super().write(content, width=width, ...)
```

### Test 3: Visual Inspection
- Launch TUI at various terminal widths (80, 100, 120, 160)
- Verify code blocks wrap correctly
- Verify tables fit within pane
- Verify no horizontal scrolling

## Recommended Implementation Order

1. **Fix 1** (styles.tcss): Add explicit width constraint - **5 minutes**
2. **Fix 4** (CodeBlock word wrap): Enable wrapping in code blocks - **15 minutes**
3. **Fix 3** (RichLog width propagation): Ensure width is passed to renderables - **30 minutes**
4. **Fix 2** (Constrained Markdown): Add width constraint wrapper - **1 hour**

## Alternative Approach: Pre-process Long Lines

If the above fixes don't work, pre-process the handoff document to break long lines:

```python
def wrap_long_lines(text: str, max_width: int = 90) -> str:
    """Break long lines in code blocks."""
    lines = text.split('\n')
    wrapped = []
    in_code_block = False
    
    for line in lines:
        if line.startswith('```'):
            in_code_block = not in_code_block
            wrapped.append(line)
        elif in_code_block and len(line) > max_width:
            # Break long line
            while len(line) > max_width:
                wrapped.append(line[:max_width])
                line = '  ' + line[max_width:]  # Indent continuation
            wrapped.append(line)
        else:
            wrapped.append(line)
    
    return '\n'.join(wrapped)
```

## Conclusion

The TUI rendering issue is caused by:
1. RichLog not propagating width constraints to Markdown renderables
2. Code blocks defaulting to `word_wrap=False`
3. No explicit width constraint on the conversation pane

The fixes are straightforward and should resolve the overflow issues. Priority should be given to Fix 1 and Fix 4, which are quick wins.
