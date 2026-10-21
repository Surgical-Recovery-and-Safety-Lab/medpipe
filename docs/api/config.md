# Configuration

---

Each class below is the schema of one section of the TOML configuration file. For a description of the sections and how to use them, see the [configuration guide](../guide/config.md).

::: medpipe.utils.config
    options:
      filters:
        - "!^_"
        - "!^validate_"
        - "!^resolve_"
        - "!^check_"
        - "!^handle_"
