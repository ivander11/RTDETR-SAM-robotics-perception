"""Expand the vendored Lite6 description into a standalone PyBullet URDF."""

import argparse
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import xacro


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def export_model(asset_dir: Path) -> Path:
    asset_dir = asset_dir.resolve()
    description = asset_dir / "upstream/xarm_description"
    source = description / "urdf/lite6/lite6.urdf.xacro"
    source_text = source.read_text()
    # Adapt only a temporary copy; preserve the vendored original verbatim.
    ros_lookup = "$(find xarm_description)"
    if source_text.count(ros_lookup) != 3:
        raise ValueError("Upstream package lookups changed; review this exporter.")
    adapted = source_text.replace(ros_lookup, "$(arg description_dir)")

    with tempfile.TemporaryDirectory(prefix="lite6-xacro-") as temp_dir:
        macro = Path(temp_dir) / "lite6.urdf.xacro"
        macro.write_text(adapted)
        document = xacro.process_file(
            str(asset_dir / "lite6_pybullet.urdf.xacro"),
            mappings={
                "description_dir": str(description),
                "lite6_macro_path": str(macro),
            },
        )

    output = asset_dir / "urdf/lite6.urdf"
    xml = document.toprettyxml(indent="  ")
    # Keep Xacro's generated-file notice reproducible across checkout locations.
    xml = xml.replace(str(asset_dir / "lite6_pybullet.urdf.xacro"),
                      "ufactory_lite6/lite6_pybullet.urdf.xacro")
    robot = ET.fromstring(xml)
    links = robot.findall("link")
    joints = robot.findall("joint")
    if len(links) != 7 or len(joints) != 6:
        raise ValueError("Expected seven links and six joints in the bare Lite6.")
    for mesh in robot.findall(".//mesh"):
        mesh_path = Path(mesh.attrib["filename"])
        if mesh_path.is_absolute() or not (output.parent / mesh_path).resolve().is_file():
            raise ValueError(f"Mesh must exist at a relative path: {mesh_path}")
    expanded_xml = ET.tostring(robot, encoding="unicode")  # Ignore source comments.
    if "$(" in expanded_xml or "${" in expanded_xml:
        raise ValueError("Unexpanded expressions remain in the URDF.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(xml)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path,
                        default=PROJECT_ROOT / "ufactory_lite6")
    args = parser.parse_args()
    print(f"Exported: {export_model(args.asset_dir)}")
