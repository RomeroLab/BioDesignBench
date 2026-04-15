"""Scripted pipeline baseline for BioDesignBench."""

from pathlib import Path
from typing import Any

from biodesignbench.agents.base import AgentInfo, AgentInterface, AgentOutput
from biodesignbench.tasks.schema import Task, TaskTier


class ScriptedPipelineBaseline(AgentInterface):
    """
    Scripted pipeline baseline - non-agentic hardcoded workflows.

    This baseline represents the lower bound for comparison.
    It uses predefined scripts for each task type without
    any LLM reasoning or adaptation.
    """

    def __init__(self):
        """Initialize scripted baseline."""
        self._templates = self._load_templates()

    def get_info(self) -> AgentInfo:
        """Return agent metadata."""
        return AgentInfo(
            agent_id="scripted-baseline",
            name="Scripted Pipeline",
            version="1.0.0",
            description="Non-agentic hardcoded workflow baseline",
            provider="baseline",
            model="none",
            is_bio_specific=False,
            capabilities=["deterministic", "scripted"],
        )

    def setup(self) -> None:
        """No setup needed for scripted baseline."""
        pass

    def teardown(self) -> None:
        """No cleanup needed for scripted baseline."""
        pass

    async def solve(self, task: Task, output_dir: Path | None = None) -> AgentOutput:
        """
        Return scripted solution for task.

        Args:
            task: Task object with description, inputs, constraints

        Returns:
            AgentOutput with pre-defined scripted solution
        """
        # Get template based on task type
        template = self._get_template(task.task_type)

        # Generate code from template
        generated_code = self._generate_code(task, template)

        # Build output based on task tier
        if task.tier == TaskTier.TIER1:
            return AgentOutput(
                code=generated_code,
                artifacts=[],
                tools_used=["scripted"],
                api_calls=0,
                iterations=1,
                reasoning_trace="Scripted baseline - no reasoning.",
            )
        else:
            # Tier 2 design tasks - scripted baseline has limited capability
            return AgentOutput(
                designs=[],
                tools_used=["scripted"],
                api_calls=0,
                iterations=1,
                reasoning_trace="Scripted baseline - no design capability.",
            )

    def _load_templates(self) -> dict[str, str]:
        """Load predefined script templates for each task type."""
        return {
            "data_retrieval": '''"""Data retrieval script."""
import requests
from Bio import SeqIO
from io import StringIO

def retrieve_data(query_params):
    """Retrieve biological data from databases."""
    # UniProt query
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query_params.get("query", "*"),
        "format": "fasta",
        "size": query_params.get("limit", 100),
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.text

if __name__ == "__main__":
    result = retrieve_data({params})
    with open("{output_file}", "w") as f:
        f.write(result)
''',
            "sequence_analysis": '''"""Sequence analysis script."""
from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
import json

def analyze_sequences(fasta_file):
    """Perform sequence analysis."""
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    results = {{
        "num_sequences": len(sequences),
        "total_length": sum(len(s) for s in sequences),
        "avg_length": sum(len(s) for s in sequences) / len(sequences) if sequences else 0,
    }}
    return results

if __name__ == "__main__":
    results = analyze_sequences("{input_file}")
    with open("{output_file}", "w") as f:
        json.dump(results, f, indent=2)
''',
            "structure_prediction": '''"""Structure prediction script."""
import requests
import json

def predict_structure(sequence):
    """Predict protein structure using ESMFold API."""
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    response = requests.post(url, data=sequence, timeout=300)
    response.raise_for_status()
    return response.text

if __name__ == "__main__":
    with open("{input_file}") as f:
        sequence = f.read().strip()
    pdb_content = predict_structure(sequence)
    with open("{output_file}", "w") as f:
        f.write(pdb_content)
''',
            "file_conversion": '''"""File conversion script."""
from Bio import SeqIO

def convert_format(input_file, input_format, output_format):
    """Convert between biological file formats."""
    records = list(SeqIO.parse(input_file, input_format))
    return records

if __name__ == "__main__":
    records = convert_format("{input_file}", "{input_format}", "{output_format}")
    SeqIO.write(records, "{output_file}", "{output_format}")
''',
            "default": '''"""Generic bioinformatics script."""
# Scripted baseline - generic template
# This task type is not fully supported by the scripted baseline.

def main():
    """Placeholder main function."""
    pass

if __name__ == "__main__":
    main()
''',
        }

    def _get_template(self, task_type: str) -> str:
        """Get template for task type."""
        return self._templates.get(task_type, self._templates["default"])

    def _generate_code(self, task: Task, template: str) -> str:
        """Generate code from template using task parameters."""
        # Extract parameters for template substitution
        params = {}

        if hasattr(task, "input_data") and task.input_data:
            if hasattr(task.input_data, "files") and task.input_data.files:
                params["input_file"] = task.input_data.files[0]
            if hasattr(task.input_data, "parameters"):
                params.update(task.input_data.parameters or {})

        if hasattr(task, "expected_output") and task.expected_output:
            if (
                hasattr(task.expected_output, "artifacts")
                and task.expected_output.artifacts
            ):
                params["output_file"] = task.expected_output.artifacts[0]

        # Default values
        params.setdefault("input_file", "input.fasta")
        params.setdefault("output_file", "output.fasta")
        params.setdefault("input_format", "fasta")
        params.setdefault("output_format", "fasta")
        params.setdefault("params", "{}")

        # Simple template substitution
        try:
            return template.format(**params)
        except KeyError:
            return template
