#!/usr/bin/env python3
"""List the controls on an Android screen that TalkBack has nothing to say for.

    python3 scripts/android_a11y_check.py DUMP.xml [DUMP.xml ...]
    python3 scripts/android_a11y_check.py --self-test

A DUMP is what `adb shell uiautomator dump` writes for one screen. A control is a node that is clickable,
long-clickable or focusable (a button, a card, a text field, a spinner). TalkBack speaks a control's own text or
content description, a text field's hint, or, for a container, its children's text. A control with none of these is
announced only as "button" or "unlabelled", and is reported here with its class, resource id and bounds.

Nodes of other packages (the system bars, the keyboard) are ignored: only --package's (default ggtts.demo).

Exit 0 when every control has a label, 1 when one does not, 2 on a usage error.
"""

import argparse
import sys
import xml.etree.ElementTree as ET

CONTROL_FLAGS = ("clickable", "long-clickable", "focusable")


def spoken(node):
    """Whether TalkBack has something to say for node: its text, content description or hint, or a child's."""
    for attr in ("text", "content-desc", "hint"):
        if (node.get(attr) or "").strip():
            return True
    return any(spoken(child) for child in node)


def unlabelled(root, package):
    """The controls of package under root with nothing to speak, as (class, resource-id, bounds)."""
    found = []
    for node in root.iter("node"):
        if node.get("package") != package:
            continue
        if not any(node.get(f) == "true" for f in CONTROL_FLAGS):
            continue
        if not spoken(node):
            found.append((node.get("class", ""), node.get("resource-id", ""), node.get("bounds", "")))
    return found


def check(paths, package):
    bad = 0
    for path in paths:
        root = ET.parse(path).getroot()
        controls = sum(
            1 for n in root.iter("node")
            if n.get("package") == package and any(n.get(f) == "true" for f in CONTROL_FLAGS)
        )
        missing = unlabelled(root, package)
        print(f"{path}: {controls} controls, {len(missing)} without a label")
        for cls, rid, bounds in missing:
            print(f"  unlabelled: {cls} {rid or '(no id)'} at {bounds}")
        bad += len(missing)
    return 1 if bad else 0


PLANTED = """<hierarchy rotation="0">
  <node class="android.widget.FrameLayout" package="ggtts.demo" clickable="false" focusable="false" bounds="[0,0][100,100]">
    <node class="android.widget.ImageButton" package="ggtts.demo" clickable="true" focusable="true" text="" content-desc="" bounds="[0,0][10,10]"/>
    <node class="android.widget.ImageButton" package="ggtts.demo" clickable="true" focusable="true" text="" content-desc="More options" bounds="[10,0][20,10]"/>
    <node class="android.widget.FrameLayout" package="ggtts.demo" clickable="true" focusable="true" text="" content-desc="" bounds="[0,20][100,40]">
      <node class="android.widget.TextView" package="ggtts.demo" clickable="false" focusable="false" text="hk_voice_343" bounds="[0,20][50,30]"/>
    </node>
    <node class="android.widget.Spinner" package="ggtts.demo" clickable="true" focusable="true" text="" content-desc="" bounds="[0,50][100,60]">
      <node class="android.widget.TextView" package="ggtts.demo" clickable="false" focusable="false" text="" bounds="[0,50][100,60]"/>
    </node>
    <node class="android.widget.EditText" package="ggtts.demo" clickable="true" focusable="true" text="" hint="What to read" bounds="[0,70][100,90]"/>
  </node>
  <node class="android.widget.FrameLayout" package="com.android.systemui" clickable="true" focusable="true" text="" bounds="[0,0][1,1]"/>
</hierarchy>"""


def self_test():
    root = ET.fromstring(PLANTED)
    found = unlabelled(root, "ggtts.demo")
    classes = sorted(c for c, _, _ in found)
    # Flagged: the bare button and the spinner whose only child says nothing.
    # Not flagged: the described button, the card whose child has text, the hinted field, another package's node.
    assert classes == ["android.widget.ImageButton", "android.widget.Spinner"], found
    assert found[0][2] == "[0,0][10,10]", found
    print("android_a11y_check: self-test passed (an unlabelled button and an empty spinner are reported; "
          "a described button, a card with text, a hinted field and other packages are not)")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dumps", nargs="*")
    p.add_argument("--package", default="ggtts.demo")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if not a.dumps:
        p.error("give one or more uiautomator dumps, or --self-test")
    return check(a.dumps, a.package)


if __name__ == "__main__":
    sys.exit(main())
