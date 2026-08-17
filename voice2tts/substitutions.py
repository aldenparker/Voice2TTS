"""Text rewriting between recognition and speech.

Two problems this solves. Whisper mishears names, gamertags and jargon it has never
seen, always in the same way -- so a rule fixes it permanently instead of every
time. And Piper pronounces some spellings badly, where writing the word out
phonetically is the only cure.

Rules run on the transcript before synthesis, so they also serve as expansions
("brb" -> "be right back") and as a way to stop the app saying something you would
rather it did not.

Matching is whole-word and case-insensitive by default, because a substring rule is
almost always a mistake: replacing "al" would maul "also", "always" and "metal".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

MAX_RULES = 500          # a pathological list would slow every utterance
MAX_PASSES = 3           # rules can feed each other; this bounds the recursion


@dataclass
class Rule:
    pattern: str
    replacement: str
    enabled: bool = True
    whole_word: bool = True
    regex: bool = False
    case_sensitive: bool = False

    def compiled(self) -> re.Pattern | None:
        """The compiled matcher, or None if the rule cannot be used."""
        if not self.pattern:
            return None
        flags = 0 if self.case_sensitive else re.IGNORECASE
        source = self.pattern if self.regex else re.escape(self.pattern)
        if self.whole_word and not self.regex:
            # \b fails next to punctuation-only patterns, so only wrap when the
            # edges are word characters.
            left = r"\b" if self.pattern[:1].isalnum() else ""
            right = r"\b" if self.pattern[-1:].isalnum() else ""
            source = f"{left}{source}{right}"
        try:
            return re.compile(source, flags)
        except re.error as exc:
            log.warning("bad substitution %r: %s", self.pattern, exc)
            return None

    def describe_error(self) -> str:
        """'' if the rule is usable, otherwise why it is not."""
        if not self.pattern:
            return "pattern is empty"
        if self.regex:
            try:
                re.compile(self.pattern)
            except re.error as exc:
                return f"invalid regular expression: {exc}"
        return ""


class Substituter:
    """Applies a rule list. Rebuilt whenever the rules change."""

    def __init__(self, rules: list[Rule] | None = None):
        self._compiled: list[tuple[re.Pattern, str]] = []
        self.load(rules or [])

    def load(self, rules: list[Rule]) -> int:
        """Compile the enabled rules. Returns how many are active."""
        self._compiled = []
        for rule in rules[:MAX_RULES]:
            if not rule.enabled:
                continue
            pattern = rule.compiled()
            if pattern is None:
                continue
            # Backslashes in a replacement are only meaningful for a regex rule,
            # where "\1" is a backreference the user meant. In a plain rule they
            # are literal text, so escape them -- otherwise typing "\1" would
            # either inject a capture group or raise "invalid group reference".
            replacement = (rule.replacement if rule.regex
                           else rule.replacement.replace("\\", "\\\\"))
            self._compiled.append((pattern, replacement))
        if len(rules) > MAX_RULES:
            log.warning("only the first %d substitutions are used", MAX_RULES)
        return len(self._compiled)

    @property
    def active(self) -> int:
        return len(self._compiled)

    def apply(self, text: str) -> str:
        """Rewrite `text`. Safe to call with no rules loaded."""
        if not text or not self._compiled:
            return text

        result = text
        for _ in range(MAX_PASSES):
            before = result
            for pattern, replacement in self._compiled:
                try:
                    result = pattern.sub(replacement, result)
                except re.error as exc:  # a bad backreference in the replacement
                    log.warning("substitution failed: %s", exc)
            if result == before:
                break  # settled; no rule fed another
        else:
            # Hit the pass limit, which means rules are rewriting each other in a
            # loop. Stop rather than spin, and say so once.
            log.warning("substitutions did not settle after %d passes", MAX_PASSES)

        if result != text:
            log.debug("substituted %r -> %r", text, result)
        return result


def preview(rules: list[Rule], sample: str) -> str:
    """Apply rules to a sample without disturbing the live substituter."""
    return Substituter(rules).apply(sample)


# Offered on first run: the expansions almost everyone wants, and an example of
# fixing a word Piper says badly.
STARTER_RULES: tuple[Rule, ...] = (
    Rule("brb", "be right back"),
    Rule("afk", "away from keyboard"),
    Rule("gg", "good game"),
    Rule("idk", "I don't know"),
    Rule("imo", "in my opinion"),
    Rule("btw", "by the way"),
    Rule("tbh", "to be honest"),
    Rule("nvm", "never mind"),
)
