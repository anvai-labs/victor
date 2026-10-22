"""Task vocabulary shared below the framework.

The task-classification protocols and the direct-response prompt classifier
are consumed by infrastructure (embedding task classifiers, classification)
as well as the runtime, so they live in core. ``victor.framework.task``
re-exports them unchanged.
"""
