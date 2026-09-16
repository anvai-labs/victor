# Advanced Docker deployment

Deployment instructions are consolidated in the [Docker guide](README.md) and
[dependency maintenance guide](../docs/development/dependencies.md#container-targets).
They describe the current image targets, offline BGE model, runtime packages,
user configuration and final-image security checks.

The former MiniLM model, precomputed tool count, bundled profile and fixed image
size descriptions are obsolete. Earlier configurations remain recoverable in Git
history. Provider servers and their model weights must be provisioned separately.
