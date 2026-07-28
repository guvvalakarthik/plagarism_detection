"""Convert PAN text-alignment truth XML into SourceLens JSONL manifests.

The script writes paths and annotations only. Copyrighted source documents stay
in the user-supplied dataset directory and must not be committed.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def truth_records(
    pairs_file: Path,
    source_directory: Path,
    candidate_directory: Path,
    truth_directory: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(
        pairs_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"invalid pairs line {line_number}")
        candidate_name, source_name = parts
        truth_path = truth_directory / f"{Path(candidate_name).stem}.xml"
        root = ET.parse(truth_path).getroot()
        annotations = []
        for feature in root.findall(".//feature[@name='plagiarism']"):
            if feature.get("source_reference") != source_name:
                continue
            candidate_start = int(feature.attrib["this_offset"])
            source_start = int(feature.attrib["source_offset"])
            annotations.append(
                {
                    "candidate_start": candidate_start,
                    "candidate_end": candidate_start + int(feature.attrib["this_length"]),
                    "source_start": source_start,
                    "source_end": source_start + int(feature.attrib["source_length"]),
                }
            )
        records.append(
            {
                "example_id": f"{candidate_name}:{source_name}",
                "group_id": source_name,
                "source_path": (source_directory / source_name).as_posix(),
                "candidate_path": (candidate_directory / candidate_name).as_posix(),
                "category": "pan_text_alignment",
                "annotations": annotations,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    records = truth_records(
        arguments.pairs,
        arguments.source_dir,
        arguments.candidate_dir,
        arguments.truth_dir,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} examples to {arguments.output}")


if __name__ == "__main__":
    main()
