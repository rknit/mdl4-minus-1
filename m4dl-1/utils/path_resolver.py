"""Path resolution utility for config paths with placeholder support."""

import re
from typing import Dict, Optional, Set


class PathResolver:
    """
    Resolves path templates with placeholder substitution.

    Supports placeholders like {disease_type}, {timestamp}, {cv_strategy}, etc.
    Validates that all placeholders are resolved and warns about unknown ones.
    """

    def __init__(self, **context_values: str):
        """
        Initialize resolver with context values for placeholder substitution.

        Args:
            **context_values: Key-value pairs for placeholder resolution
                             (e.g., disease_type="T2D", timestamp="20251027_143022")

        Example:
            resolver = PathResolver(
                disease_type="T2D",
                timestamp="20251027_143022",
                cv_strategy="loocv"
            )
        """
        self.context = {k: str(v) for k, v in context_values.items()}
        self._placeholder_pattern = re.compile(r'\{([^}]+)\}')

    def resolve(self, path_template: str, strict: bool = False) -> str:
        """
        Resolve all placeholders in a path template.

        Args:
            path_template: Path string with placeholders (e.g., "./data/{disease_type}")
            strict: If True, raises ValueError for unresolved placeholders.
                   If False (default), prints warning and leaves unresolved.

        Returns:
            Resolved path string

        Raises:
            ValueError: If strict=True and placeholders remain unresolved

        Example:
            resolver.resolve("./logs/{disease_type}/{timestamp}")
            # Returns: "./logs/T2D/20251027_143022"
        """
        resolved = path_template

        for placeholder in self._find_placeholders(path_template):
            pattern = f"{{{placeholder}}}"
            if placeholder in self.context:
                resolved = resolved.replace(pattern, self.context[placeholder])
            elif strict:
                raise ValueError(
                    f"Unresolved placeholder '{placeholder}' in path: {path_template}\n"
                    f"Available context: {list(self.context.keys())}"
                )
            else:
                print(
                    f"WARNING: Placeholder '{placeholder}' not found in context. "
                    f"Available: {list(self.context.keys())}"
                )

        return resolved

    def resolve_all(
        self,
        path_templates: Dict[str, str],
        strict: bool = False
    ) -> Dict[str, str]:
        """
        Resolve multiple path templates at once.

        Args:
            path_templates: Dictionary of {name: path_template}
            strict: If True, raises ValueError for unresolved placeholders

        Returns:
            Dictionary of {name: resolved_path}

        Example:
            templates = {
                "data_dir": "./data/{disease_type}",
                "logs_dir": "./logs/{disease_type}/{timestamp}"
            }
            resolved = resolver.resolve_all(templates)
        """
        return {
            name: self.resolve(template, strict=strict)
            for name, template in path_templates.items()
        }

    def add_context(self, **new_context: str) -> None:
        """
        Add or update context values for placeholder resolution.

        Args:
            **new_context: Additional key-value pairs to add to context

        Example:
            resolver.add_context(modality_name="modality_1", fold="0")
        """
        self.context.update({k: str(v) for k, v in new_context.items()})

    def get_context(self) -> Dict[str, str]:
        """Return a copy of the current context dictionary."""
        return self.context.copy()

    def validate_template(self, path_template: str) -> tuple[bool, Set[str]]:
        """
        Check if a path template can be fully resolved with current context.

        Args:
            path_template: Path string with placeholders

        Returns:
            Tuple of (is_valid, set_of_missing_placeholders)

        Example:
            is_valid, missing = resolver.validate_template("./data/{disease_type}/{fold}")
            if not is_valid:
                print(f"Missing placeholders: {missing}")
        """
        placeholders = self._find_placeholders(path_template)
        missing = placeholders - set(self.context.keys())
        return len(missing) == 0, missing

    def _find_placeholders(self, path_template: str) -> Set[str]:
        """Extract all placeholder names from a path template."""
        return set(self._placeholder_pattern.findall(path_template))


def create_resolver_from_config(
    disease_type: str,
    timestamp: str,
    cv_strategy: Optional[str] = None,
    **extra_context: str
) -> PathResolver:
    """
    Convenience factory for creating PathResolver from common config values.

    Args:
        disease_type: Disease type (e.g., "T2D", "CRC")
        timestamp: Timestamp string (e.g., "20251027_143022")
        cv_strategy: CV strategy (e.g., "loocv", "kfold") - optional
        **extra_context: Additional context values

    Returns:
        Configured PathResolver instance

    Example:
        resolver = create_resolver_from_config(
            disease_type="T2D",
            timestamp="20251027_143022",
            cv_strategy="loocv"
        )
    """
    context = {
        "disease_type": disease_type,
        "timestamp": timestamp,
    }

    if cv_strategy is not None:
        context["cv_strategy"] = cv_strategy

    context.update(extra_context)

    return PathResolver(**context)
