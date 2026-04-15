# BioDesignBench - Product Requirements Document

**Version**: 0.1.0
**Last Updated**: January 2026
**Status**: Draft

---

## 1. Executive Summary

BioDesignBench는 biomolecule design을 위한 AI 에이전트를 평가하는 최초의 comprehensive benchmark이다. 기존 벤치마크들이 model-only 평가(ProteinBench, AbBiBench) 또는 analysis 중심 agent 평가(BioML-bench, Biomni)에 집중한 반면, BioDesignBench는 **"Natural language → Design → Evaluate → Iterate"** full loop을 평가한다.

---

## 2. Goals & Non-Goals

### 2.1 Goals

- **G1**: Bio coding task와 in silico design task를 포함하는 2-tier 벤치마크 구축
- **G2**: General-purpose LLM agents와 Bio-specific agents를 동일 기준으로 비교
- **G3**: 재현 가능하고 자동화된 평가 파이프라인 제공
- **G4**: Data contamination을 방지하는 평가 환경 구축

### 2.2 Non-Goals

- Wet-lab 실험 연동 (Tier 3는 향후 확장)
- 실시간 leaderboard 운영 (초기 버전에서는 제외)
- 모든 biomolecule 타입 커버 (단백질 중심으로 시작)

---

## 3. Scope

### 3.1 Tier 1: Bio Coding Agent Benchmark

**목표**: Bioinformatics pipeline automation 능력 평가

**Task 예시**:

- Sequence retrieval & filtering (UniProt, PDB)
- Structure prediction pipeline setup (ESMFold, AlphaFold2)
- Variant library generation
- MSA construction & analysis
- Molecular dynamics simulation setup

**평가 방식**: Unit test 스타일 (pass/fail + partial credit)

### 3.2 Tier 2: In Silico Design Benchmark

**목표**: Computational design quality 평가

**Task 예시**:

- De novo binder design (BindCraft, AlphaProteo 스타일)
- CDR optimization for antibodies
- Enzyme active site redesign
- Stability optimization
- Humanization

**평가 방식**: 4D metrics (Quality, Novelty, Diversity, Feasibility)

---

## 4. Target Agents (비교 대상)

### 4.1 General-Purpose LLM Agents

| Agent            | Provider    | Notes             |
| ---------------- | ----------- | ----------------- |
| Claude Code      | Anthropic   | Claude + tool use |
| GPT-4 + Tools    | OpenAI      | Function calling  |
| Gemini + Tools   | Google      | Code execution    |
| Open Interpreter | Open Source | Local execution   |

### 4.2 Bio-Specific Agents

| Agent       | Source | Specialty                   |
| ----------- | ------ | --------------------------- |
| Biomni      | NYU    | Multi-modal bio agent       |
| STELLA      | -      | Scientific literature agent |
| BioML-agent | -      | Biomedical ML               |

### 4.3 Baselines

| Baseline          | Description                      |
| ----------------- | -------------------------------- |
| Scripted Pipeline | Non-agentic, hardcoded workflow  |
| Human Expert      | Published results from papers    |
| Human Trainee     | Graduate student (1 week effort) |

---

## 5. Task Sources

### 5.1 Retrospective Tasks (Ground Truth 있음)

최신 논문(2024-2025)에서 추출, 에이전트에게는 논문 출판 이전 날짜로 검색 제한

| Source            | Task Type             | Count (Target) |
| ----------------- | --------------------- | -------------- |
| BindCraft paper   | Binder design         | 5-10           |
| AlphaProteo paper | Binder design         | 5-10           |
| Boltz paper       | Structure prediction  | 5-10           |
| Baker Lab papers  | Various design        | 10-20          |
| AbBiBench dataset | Antibody optimization | 10-20          |

### 5.2 Novel Tasks (Ground Truth 없음)

새로운 target에 대한 design, predictive metrics로만 평가

| Task Type    | Description                 |
| ------------ | --------------------------- |
| Novel binder | 기존에 binder가 없는 target |
| Optimization | 기존 sequence 개선          |

### 5.3 Bio Coding Tasks

| Category          | Task Examples                | Count (Target) |
| ----------------- | ---------------------------- | -------------- |
| Data Retrieval    | UniProt query, PDB download  | 10             |
| Analysis Pipeline | MSA, phylogenetic tree       | 10             |
| Prediction Setup  | AF2, ESMFold execution       | 10             |
| File Conversion   | PDB ↔ FASTA, format handling | 5              |

---

## 6. Task I/O Specification

### 6.1 Tier 1: Bio Coding Task Format

```json
{
  "task_id": "biocoding_001",
  "task_type": "coding",
  "tier": 1,
  "description": "Task description in natural language",
  "input_data": {
    "files": ["path/to/input.fasta"],
    "parameters": {}
  },
  "expected_output": {
    "format": "python_script",
    "artifacts": ["output.fasta", "results.json"]
  },
  "evaluation": {
    "method": "unit_test",
    "test_file": "tests/test_biocoding_001.py"
  },
  "constraints": {
    "time_limit_minutes": 30,
    "knowledge_cutoff": "2023-06-01"
  },
  "metadata": {
    "difficulty": "medium",
    "tools_expected": ["biopython", "requests"],
    "source": "synthetic"
  }
}
```

### 6.2 Tier 2: Design Task Format

```json
{
  "task_id": "design_001",
  "task_type": "binder_design",
  "tier": 2,
  "description": "Design a de novo protein binder for human IL-17A",
  "target": {
    "name": "IL-17A",
    "pdb_id": "4HR9",
    "sequence": "GITIPRNPGC...",
    "binding_site_residues": [23, 45, 67]
  },
  "constraints": {
    "length_range": [50, 80],
    "excluded_residues": ["C"],
    "max_designs": 10
  },
  "expected_output": {
    "format": "design_bundle",
    "required_files": [
      "designed_sequences.fasta",
      "predicted_structures/*.pdb",
      "metrics.json"
    ]
  },
  "evaluation": {
    "metrics": ["pLDDT", "ipTM", "predicted_kd", "novelty", "diversity"],
    "ground_truth": {
      "known_binder_sequence": "MKTL...",
      "experimental_kd_nM": 2.3
    }
  },
  "constraints": {
    "time_limit_minutes": 120,
    "knowledge_cutoff": "2024-01-01"
  },
  "metadata": {
    "difficulty": "hard",
    "source": "BindCraft paper",
    "doi": "10.1038/..."
  }
}
```

### 6.3 Output Format (metrics.json)

```json
{
  "task_id": "design_001",
  "agent_id": "claude-code-v1",
  "timestamp": "2026-01-12T10:00:00Z",
  "designs": [
    {
      "id": "design_001_seq1",
      "sequence": "MKTLLILAVV...",
      "length": 65,
      "quality": {
        "pLDDT": 87.3,
        "ipTM": 0.82,
        "predicted_kd_nM": 5.2
      },
      "novelty": {
        "max_sequence_identity": 0.34,
        "max_tm_score_to_pdb": 0.45
      }
    }
  ],
  "execution": {
    "total_time_seconds": 3600,
    "api_calls": 150,
    "tools_used": ["esmfold", "proteinmpnn", "alphafold2"],
    "iterations": 3
  },
  "summary": {
    "total_designs": 5,
    "valid_designs": 4,
    "best_predicted_kd_nM": 5.2,
    "mean_pLDDT": 83.1
  }
}
```

---

## 7. Evaluation Metrics

### 7.1 Tier 1 Metrics (Bio Coding)

| Metric                         | Description                    | Range   |
| ------------------------------ | ------------------------------ | ------- |
| **Valid Execution Rate (VER)** | 코드가 에러 없이 실행되는 비율 | 0-100%  |
| **Success Rate (SR)**          | 정답을 생성하는 비율           | 0-100%  |
| **Partial Score**              | 부분 점수 (rubric 기반)        | 0-100   |
| **Time Efficiency**            | 완료 시간                      | seconds |
| **Cost**                       | API 호출 비용                  | USD     |

### 7.2 Tier 2 Metrics (Design)

| Dimension       | Metrics                                  | Description      |
| --------------- | ---------------------------------------- | ---------------- |
| **Quality**     | pLDDT, ipTM, predicted_Kd                | 디자인 품질      |
| **Novelty**     | max_seq_identity, max_TM_to_PDB          | 기존 대비 새로움 |
| **Diversity**   | pairwise_TM, cluster_count               | 디자인 다양성    |
| **Feasibility** | aggregation_score, expression_likelihood | 실현 가능성      |

### 7.3 Agent Capability Metrics

| Metric                   | Description                            |
| ------------------------ | -------------------------------------- |
| **Iteration Efficiency** | (Final - Initial quality) / iterations |
| **Feedback Utilization** | 피드백 반영 비율                       |
| **Tool Coverage**        | 사용한 도구 / 가용 도구                |
| **Error Recovery**       | 에러 발생 시 복구 능력                 |

---

## 8. Implementation Phases

### Phase 1: Task Collection (Month 1-2)

**목표**: 50-70개 task 수집

| Step | Description                     | Output                  |
| ---- | ------------------------------- | ----------------------- |
| 1.1  | Task taxonomy 정의              | `docs/task_taxonomy.md` |
| 1.2  | Bio coding tasks 설계 (35개)    | `tasks/tier1/*.json`    |
| 1.3  | Design tasks 수집 (15-35개)     | `tasks/tier2/*.json`    |
| 1.4  | 논문에서 ground truth 추출      | `data/ground_truth/`    |
| 1.5  | Task validation (expert review) | Review checklist        |

**Deliverables**:

- Task JSON files
- Input data files
- Ground truth data (retrospective tasks)

### Phase 2: Metrics Implementation (Month 2-3)

**목표**: 자동화된 평가 시스템 구축

| Step | Description                                 | Output                            |
| ---- | ------------------------------------------- | --------------------------------- |
| 2.1  | Tier 1 test harness 구현                    | `biodesignbench/eval/tier1/`      |
| 2.2  | Tier 2 metrics 구현                         | `biodesignbench/eval/tier2/`      |
| 2.3  | Structure prediction wrapper (AF2, ESMFold) | `biodesignbench/tools/`           |
| 2.4  | Scoring aggregation                         | `biodesignbench/eval/scorer.py`   |
| 2.5  | Evaluation pipeline 통합                    | `biodesignbench/eval/pipeline.py` |

**Deliverables**:

- Evaluation harness code
- Metric computation functions
- Integration tests

### Phase 3: Baseline Construction (Month 3-4)

**목표**: 비교 기준점 확립

| Step | Description                            | Output                  |
| ---- | -------------------------------------- | ----------------------- |
| 3.1  | Scripted baseline 구현                 | `baselines/scripted/`   |
| 3.2  | General LLM agent 연동 (Claude, GPT-4) | `baselines/llm_agents/` |
| 3.3  | Bio-specific agent 연동 (Biomni 등)    | `baselines/bio_agents/` |
| 3.4  | Human baseline 수집                    | `baselines/human/`      |
| 3.5  | Baseline 결과 기록                     | `results/baselines/`    |

**Deliverables**:

- Agent wrapper implementations
- Baseline run scripts
- Human baseline data

### Phase 4: Benchmark Execution (Month 4-5)

**목표**: 전체 벤치마크 실행

| Step | Description                  | Output                 |
| ---- | ---------------------------- | ---------------------- |
| 4.1  | Sandbox 환경 구축 (Docker)   | `docker/`              |
| 4.2  | Knowledge cutoff 적용 시스템 | Date-restricted search |
| 4.3  | 전체 agent 대상 실행         | `results/runs/`        |
| 4.4  | 결과 수집 및 검증            | `results/validated/`   |
| 4.5  | 재현성 검증 (3회 실행)       | Variance analysis      |

**Deliverables**:

- Docker images
- Run scripts
- Raw results

### Phase 5: Analysis & Publication (Month 5-6)

**목표**: 결과 분석 및 논문 작성

| Step | Description            | Output                         |
| ---- | ---------------------- | ------------------------------ |
| 5.1  | 결과 통계 분석         | `analysis/statistics/`         |
| 5.2  | Visualization          | `analysis/figures/`            |
| 5.3  | Agent별 강점/약점 분석 | `analysis/agent_comparison.md` |
| 5.4  | 논문 초안 작성         | `paper/draft.tex`              |
| 5.5  | 코드/데이터 공개 준비  | GitHub release                 |

**Deliverables**:

- Analysis notebooks
- Publication figures
- Paper draft
- Public release

---

## 9. Technical Architecture

### 9.1 Directory Structure

```
BioDesignBench/
├── biodesignbench/           # Main package
│   ├── tasks/                # Task loading & management
│   ├── eval/                 # Evaluation harness
│   │   ├── tier1/           # Bio coding evaluators
│   │   ├── tier2/           # Design evaluators
│   │   └── metrics/         # Metric implementations
│   ├── agents/              # Agent interfaces
│   ├── tools/               # Bio tool wrappers
│   └── utils/               # Utilities
├── tasks/                    # Task definitions
│   ├── tier1/               # Bio coding tasks
│   └── tier2/               # Design tasks
├── data/                     # Input data & ground truth
├── baselines/               # Baseline implementations
├── results/                 # Benchmark results
├── tests/                   # Test suite
├── docker/                  # Container definitions
├── docs/                    # Documentation
└── scripts/                 # Utility scripts
```

### 9.2 Key Dependencies

| Category                 | Tools                                |
| ------------------------ | ------------------------------------ |
| **Structure Prediction** | AlphaFold2, ESMFold, Boltz           |
| **Design Tools**         | ProteinMPNN, RFdiffusion, LigandMPNN |
| **Analysis**             | BioPython, PyMOL, MDAnalysis         |
| **ML Framework**         | PyTorch, JAX                         |
| **Evaluation**           | TMalign, Foldseek                    |

### 9.3 Agent Interface

```python
class AgentInterface(ABC):
    """Base interface for all agents."""

    @abstractmethod
    def solve(self, task: Task) -> AgentOutput:
        """
        Solve a benchmark task.

        Args:
            task: Task object with description, inputs, constraints

        Returns:
            AgentOutput with results, metrics, execution trace
        """
        pass

    @abstractmethod
    def get_info(self) -> AgentInfo:
        """Return agent metadata."""
        pass
```

---

## 10. Risk & Mitigation

| Risk                   | Impact        | Mitigation                                 |
| ---------------------- | ------------- | ------------------------------------------ |
| Data contamination     | 평가 무효화   | Knowledge cutoff 강제, novel tasks 포함    |
| Tool dependency 복잡성 | 재현성 저하   | Docker 컨테이너화, 버전 고정               |
| Metric validity        | 실험과 불일치 | 기존 validated metrics 사용, expert review |
| Agent API 변경         | 유지보수 부담 | Abstraction layer, version pinning         |
| Compute cost           | 예산 초과     | Tiered evaluation, caching                 |

---

## 11. Success Criteria

### 11.1 Minimum Viable Benchmark

- [ ] 50개 이상의 validated tasks
- [ ] 3개 이상의 agent 평가 완료
- [ ] 자동화된 evaluation pipeline
- [ ] 재현 가능한 결과

### 11.2 Publication Ready

- [ ] 70개 이상의 tasks
- [ ] 5개 이상의 agent 비교
- [ ] Human baseline 포함
- [ ] Comprehensive analysis
- [ ] Public code & data release

---

## 12. Target Venues

| Venue                         | Deadline | Notes          |
| ----------------------------- | -------- | -------------- |
| NeurIPS Datasets & Benchmarks | May 2026 | Primary target |
| ICLR                          | Sep 2026 | Alternative    |
| Nature Methods                | Rolling  | High impact    |

---

## Appendix A: Related Work

| Benchmark    | Focus           | Agentic | Design  |
| ------------ | --------------- | ------- | ------- |
| ProteinBench | Protein models  | No      | Yes     |
| AbBiBench    | Antibody models | No      | Yes     |
| BioML-bench  | Bio ML agents   | Yes     | Partial |
| Biomni       | Bio agents      | Yes     | No      |
| SWE-bench    | Code agents     | Yes     | No      |
| MLE-bench    | ML agents       | Yes     | No      |

---

## Appendix B: Glossary

| Term     | Definition                                                |
| -------- | --------------------------------------------------------- |
| pLDDT    | Predicted Local Distance Difference Test (AF2 confidence) |
| ipTM     | Interface predicted TM-score                              |
| TM-score | Template Modeling score (structure similarity)            |
| CDR      | Complementarity-Determining Region (antibody)             |
| Kd       | Dissociation constant (binding affinity)                  |
